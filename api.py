import asyncio
import warnings
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Literal, Optional
from pathlib import Path

from brain.fast_search import initialize_bm25
from brain.ingest import ingest_docs
from brain.ingest import ingest_file
from brain.augmented_generation_query import query_brain_comprehensive, session_chat_history
from brain.pdf_utils import load_pdfs
from brain.config import DATA_DIR

warnings.filterwarnings("ignore")

raw_docs = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global raw_docs
    raw_docs = load_pdfs(DATA_DIR)
    initialize_bm25(raw_docs)
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    question: str
    verbose: bool = False
    mode: Literal['auto', 'fast', 'deep', 'deep_semantic', 'both'] = 'auto'


class FeedbackRequest(BaseModel):
    question: str
    prev_mode: Optional[Literal['auto', 'fast', 'deep', 'deep_semantic', 'both']] = 'auto'

@app.post("/query")
async def query(req: QueryRequest):
    # pass along mode override; query_brain_comprehensive handles 'both'
    results = await query_brain_comprehensive(
        req.question,
        verbose=req.verbose,
        raw_docs=raw_docs,
        session_chat_history=session_chat_history,
        mode_override=req.mode
    )
    return results

@app.post("/ingest")
async def ingest():
    print("[API] ingest called")
    global raw_docs
    docs, topic_synonyms = await ingest_docs()
    raw_docs = load_pdfs(DATA_DIR)
    return {"topics": list(topic_synonyms.keys())}


@app.post("/open_data_folder")
async def open_data_folder():
    print("[API] open_data_folder called")
    import os, subprocess, sys
    data_dir = DATA_DIR

    p = Path(str(data_dir))
    if not p.exists():
        return {"status": "error", "error": f"path not found: {p}"}
    try:
        if sys.platform.startswith('win'):
            try:
                os.startfile(str(p))
            except Exception as e_start:
                # fallback to explorer.exe
                try:
                    rc = subprocess.run(['explorer', str(p)], check=False)
                    return {"status": "opened_fallback", "path": str(p), "rc": rc.returncode, "note": "used explorer fallback", "start_error": str(e_start)}
                except Exception as e_ex:
                    return {"status": "error", "error": str(e_ex), "start_error": str(e_start), "platform": sys.platform}
        elif sys.platform.startswith('darwin'):
            subprocess.run(['open', str(p)], check=True)
        else:
            subprocess.run(['xdg-open', str(p)], check=True)
        return {"status": "opened", "path": str(p), "platform": sys.platform}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {"status": "error", "error": str(e), "trace": tb, "platform": sys.platform, "path": str(p), "exists": p.exists(), "is_dir": p.is_dir(), "cwd": str(Path.cwd())}


@app.post('/upload_and_ingest')
async def upload_and_ingest(file: UploadFile = File(...)):
    # Accepts multipart file upload (form field 'file') and saves into DATA_DIR
    print("[API] upload_and_ingest called")
    try:
        upload = file
        filename = Path(upload.filename).name
        dest = Path(DATA_DIR) / filename
        
        if dest.exists():
            return {"status": "exists", "filename": filename}
        contents = await upload.read()
        with dest.open('wb') as f:
            f.write(contents)

        result = await ingest_file(str(dest))
        if result is None:
            return {"status": "exists", "filename": filename}
        docs, topics = result
        return {"status": "ingested", "filename": filename, "topics": list(topics.keys())}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post('/ingest_file')
async def ingest_file_endpoint(filename: str):
    print("[API] ingest_file called; filename=", filename)
    # filename relative to DATA_DIR
    try:
        result = await ingest_file(filename)
        if result is None:
            return {"status": "exists"}
        docs, topics = result
        return {"status": "ingested", "topics": list(topics.keys())}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    # User disliked result - escalate to semantic reranking with cross-encoder
    results = await query_brain_comprehensive(
        req.question,
        verbose=False,
        raw_docs=raw_docs,
        session_chat_history=session_chat_history,
        mode_override='deep_semantic'
    )
    return results


class AgentEditRequest(BaseModel):
    instruction: str
    file_path: str


@app.post("/agent/edit")
async def agent_edit(req: AgentEditRequest):
    """Code agent endpoint for dry-run preview."""
    from agent.code_agent import CodeAgent
    from pathlib import Path

    if not req.file_path:
        return {"error": "No file path provided", "status": "error"}

    # Resolve path: if not found as-is, search repo for a file with that name
    resolved_path = req.file_path
    if not Path(resolved_path).exists():
        # Try relative to Aion root (cwd)
        candidate = Path(resolved_path)
        if not candidate.is_absolute():
            # already tried relative, now search recursively
            matches = list(Path(".").rglob(candidate.name))
            if matches:
                resolved_path = str(matches[0].resolve())
            else:
                return {"error": f"File not found: {req.file_path}", "status": "error",
                        "message": f"File not found: {req.file_path}. Please provide the full absolute path."}
        else:
            return {"error": f"File not found: {req.file_path}", "status": "error",
                    "message": f"File not found: {req.file_path}"}

    try:
        agent = CodeAgent(repo_path=".")
        result = agent.edit_code(
            path=resolved_path,
            instruction=req.instruction,
            dry_run=True,
            use_rag=True,
            rerank_method="auto"
        )
        # edit_code returns a dict when dry_run=True: {new_source, diff, changed}
        if isinstance(result, dict):
            diff = result.get("diff", "")
            changed = result.get("changed", False)
            dry_run_output = diff if (changed and diff) else "(No changes — agent could not determine what to modify.)"
        else:
            dry_run_output = str(result)
        return {
            "status": "pending_review",
            "dry_run_output": dry_run_output,
            "file_path": resolved_path
        }
    except Exception as e:
        return {"error": str(e), "status": "error", "message": f"Agent error: {str(e)}"}


class AgentApplyRequest(BaseModel):
    instruction: str
    file_path: str
    confirmed: bool = False


@app.post("/agent/apply")
async def agent_apply(req: AgentApplyRequest):
    """Apply code agent changes to file."""
    from agent.code_agent import CodeAgent
    from pathlib import Path

    if not req.confirmed:
        return {"error": "Changes not confirmed", "status": "error"}

    # Resolve path the same way as /agent/edit
    resolved_path = req.file_path
    if not Path(resolved_path).exists():
        candidate = Path(resolved_path)
        if not candidate.is_absolute():
            matches = list(Path(".").rglob(candidate.name))
            if matches:
                resolved_path = str(matches[0].resolve())
            else:
                return {"error": f"File not found: {req.file_path}", "status": "error",
                        "message": f"File not found: {req.file_path}"}
        else:
            return {"error": f"File not found: {req.file_path}", "status": "error",
                    "message": f"File not found: {req.file_path}"}

    try:
        agent = CodeAgent(repo_path=".")
        result = agent.edit_code(
            path=resolved_path,
            instruction=req.instruction,
            dry_run=False,
            use_rag=True,
            rerank_method="cross_encoder"  # always use best quality when actually writing
        )
        return {
            "status": "success",
            "message": f"Changes applied to {resolved_path}",
            "result": result
        }
    except Exception as e:
        return {"error": str(e), "status": "error", "message": f"Failed to apply changes: {str(e)}"}


class AgentEditWithChunksRequest(BaseModel):
    instruction: str
    file_path: str
    max_chunks: int = 7


@app.post("/agent/edit_with_chunks")
async def agent_edit_with_chunks(req: AgentEditWithChunksRequest):
    """Code agent endpoint with custom RAG chunk limit."""
    from agent.code_agent import CodeAgent
    from pathlib import Path

    if not req.file_path:
        return {"error": "No file path provided", "status": "error"}

    # Resolve path
    resolved_path = req.file_path
    if not Path(resolved_path).exists():
        candidate = Path(resolved_path)
        if not candidate.is_absolute():
            matches = list(Path(".").rglob(candidate.name))
            if matches:
                resolved_path = str(matches[0].resolve())
            else:
                return {"error": f"File not found: {req.file_path}", "status": "error"}
        else:
            return {"error": f"File not found: {req.file_path}", "status": "error"}

    try:
        # Clamp max_chunks to reasonable range
        max_chunks = max(1, min(req.max_chunks, 50))

        agent = CodeAgent(repo_path=".")
        result = agent.edit_code(
            path=resolved_path,
            instruction=req.instruction,
            dry_run=True,
            use_rag=True,
            rerank_method="auto",
            max_chunks=max_chunks
        )

        if isinstance(result, dict):
            diff = result.get("diff", "")
            changed = result.get("changed", False)
            dry_run_output = diff if (changed and diff) else "(No changes — agent could not determine what to modify.)"
        else:
            dry_run_output = str(result)

        return {
            "status": "pending_review",
            "dry_run_output": dry_run_output,
            "file_path": resolved_path,
            "max_chunks": max_chunks
        }
    except Exception as e:
        return {"error": str(e), "status": "error", "message": f"Agent error with {req.max_chunks} chunks: {str(e)}"}


@app.post("/clear")
async def clear():
    session_chat_history.clear()
    return {"status": "cleared"}
