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
from brain.chat_session_store import ChatSessionStore
from brain.pdf_utils import load_pdfs
from brain.config import DATA_DIR

warnings.filterwarnings("ignore")

raw_docs = []
chat_store = ChatSessionStore()

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
    session_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    question: str
    prev_mode: Optional[Literal['auto', 'fast', 'deep', 'deep_semantic', 'both']] = 'auto'
    session_id: Optional[str] = None

@app.post("/query")
async def query(req: QueryRequest):
    # Build context from persistent session if session_id provided
    if req.session_id:
        history = chat_store.get_context_window(req.session_id, req.question)
    else:
        history = session_chat_history

    results = await query_brain_comprehensive(
        req.question,
        verbose=req.verbose,
        raw_docs=raw_docs,
        session_chat_history=history,
        mode_override=req.mode
    )

    # Persist turns to Chroma session store
    if req.session_id:
        chat_store.add_turn(req.session_id, "User", req.question)
        answer_text = ""
        if isinstance(results, dict):
            if "deep" in results:
                answer_text = results["deep"].get("answer", "")
            else:
                answer_text = results.get("answer", "")
        if answer_text:
            chat_store.add_turn(req.session_id, "Assistant", answer_text)
        # Trigger smart summarization when chat gets long
        if chat_store.should_summarize(req.session_id):
            chat_store.trigger_summarization(req.session_id)

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
    # Tiered escalation: fast → deep → deep_semantic
    escalation = {
        'fast': 'deep',
        'auto': 'deep',
        'deep': 'deep_semantic',
        'deep_semantic': 'deep_semantic',  # already highest, retry same
        'both': 'deep_semantic',
    }
    next_mode = escalation.get(req.prev_mode or 'auto', 'deep')

    if req.session_id:
        history = chat_store.get_context_window(req.session_id, req.question)
    else:
        history = session_chat_history

    results = await query_brain_comprehensive(
        req.question,
        verbose=False,
        raw_docs=raw_docs,
        session_chat_history=history,
        mode_override=next_mode
    )

    # Persist the retry answer
    if req.session_id:
        answer_text = results.get("answer", "") if isinstance(results, dict) else ""
        if answer_text:
            chat_store.add_turn(req.session_id, "Assistant", f"[DEEP RETRY] {answer_text}")

    return results


class AgentEditRequest(BaseModel):
    instruction: str
    file_path: str
    task_mode: str = "auto"  # "fix", "solve", or "auto"
    session_id: Optional[str] = None
    context_files: list = []  # additional file paths for cross-file context


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
        # Build cross-file context from additional context files
        augmented_instruction = req.instruction
        if req.context_files:
            context_block = "\n\nRELATED FILES (read-only context):\n"
            for cf in req.context_files[:10]:
                cf_path = Path(cf)
                if cf_path.exists() and cf_path.is_file():
                    try:
                        content = cf_path.read_text(encoding="utf-8", errors="replace")[:3000]
                        context_block += f"\n--- {cf} ---\n{content}\n"
                    except Exception:
                        pass
            augmented_instruction += context_block

        agent = CodeAgent(repo_path=".")
        result = agent.edit_code(
            path=resolved_path,
            instruction=augmented_instruction,
            dry_run=True,
            use_rag=True,
            rerank_method="auto",
            task_mode=req.task_mode,
            session_id=req.session_id,
        )
        # edit_code returns a dict when dry_run=True: {new_source, diff, changed}
        if isinstance(result, dict):
            diff = result.get("diff", "")
            changed = result.get("changed", False)
            dry_run_output = diff if (changed and diff) else "(No changes — agent could not determine what to modify.)"
        else:
            dry_run_output = str(result)
        explanation = ""
        citations = []
        if isinstance(result, dict):
            explanation = result.get("explanation", "")
            citations = result.get("citations", [])
        
        # Store agent interaction into the shared query chat history
        # so follow-up questions in query mode have context
        session_chat_history.append({"role": "User", "content": req.instruction})
        if explanation:
            session_chat_history.append({"role": "Assistant", "content": explanation})

        # Also persist to chat session store for cross-restart memory
        if req.session_id:
            chat_store.add_turn(req.session_id, "User", f"[AGENT] {req.instruction}")
            if explanation:
                chat_store.add_turn(req.session_id, "Assistant", explanation)
        
        return {
            "status": "pending_review",
            "dry_run_output": dry_run_output,
            "file_path": resolved_path,
            "explanation": explanation,
            "citations": citations,
        }
    except Exception as e:
        return {"error": str(e), "status": "error", "message": f"Agent error: {str(e)}"}


class AgentApplyRequest(BaseModel):
    instruction: str
    file_path: str
    confirmed: bool = False
    task_mode: str = "auto"  # "fix", "solve", or "auto"
    session_id: Optional[str] = None


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
            rerank_method="cross_encoder",  # always use best quality when actually writing
            task_mode=req.task_mode,
            session_id=req.session_id,
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
    task_mode: str = "auto"  # "fix", "solve", or "auto"
    search_method: str = "both"  # "bm25", "semantic", or "both"


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
            max_chunks=max_chunks,
            task_mode=req.task_mode,
            search_method=req.search_method
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


# ── Multi-Agent orchestration endpoint ─────────────────────────────────────────

class OrchestrateRequest(BaseModel):
    instruction: str
    file_path: str
    task_mode: str = "auto"
    session_id: Optional[str] = None
    max_attempts: int = 3
    include_related: bool = True   # multi-file context
    context_files: list = []       # additional file paths for cross-file context


@app.post("/agent/orchestrate")
async def agent_orchestrate(req: OrchestrateRequest):
    """
    Multi-agent pipeline: Planner → Code Agent → Critic (with retry loop).
    Also gathers related files for cross-file context.
    """
    from agent.multi_agent_graph import run_multi_agent
    from pathlib import Path

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
        result = run_multi_agent(
            instruction=req.instruction,
            file_path=resolved_path,
            task_mode=req.task_mode,
            session_id=req.session_id,
            max_attempts=max(1, min(req.max_attempts, 5)),
            include_related=req.include_related,
            extra_context_files=req.context_files,
        )

        # Persist agent conversation to chat session store
        if req.session_id:
            chat_store.add_turn(req.session_id, "User", f"[AGENT] {req.instruction}")
            explanation = result.get("explanation", "")
            if explanation:
                chat_store.add_turn(req.session_id, "Assistant", explanation)
        # Also push to in-memory history for query follow-ups
        session_chat_history.append({"role": "User", "content": req.instruction})
        if result.get("explanation"):
            session_chat_history.append({"role": "Assistant", "content": result["explanation"]})

        return result
    except Exception as e:
        return {"error": str(e), "status": "error"}


# ── Test runner endpoints ──────────────────────────────────────────────────────

class RunTestsRequest(BaseModel):
    file_path: str
    test_cases: list  # [{input: str, expected: str}]


@app.post("/agent/run_tests")
async def run_tests_endpoint(req: RunTestsRequest):
    """Run test cases against the current state of a file and return pass/fail results."""
    from agent.test_runner import run_tests
    from agent.tools import read_file as agent_read_file
    from pathlib import Path

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
        source = agent_read_file(resolved_path)
        results = run_tests(source, resolved_path, req.test_cases)
        return {
            "status": "ok",
            "results": results,
            "all_passed": all(r["passed"] for r in results),
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}


class FixWithTestsRequest(BaseModel):
    file_path: str
    instruction: str
    test_cases: list  # [{input: str, expected: str}]
    max_retries: int = 3
    task_mode: str = "solve"
    session_id: Optional[str] = None


@app.post("/agent/fix_with_tests")
async def fix_with_tests_endpoint(req: FixWithTestsRequest):
    """Iteratively fix a file until all test cases pass or max_retries is exhausted."""
    from agent.code_agent import CodeAgent
    from pathlib import Path

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
        agent = CodeAgent(repo_path=".")
        result = agent.fix_with_tests(
            path=resolved_path,
            instruction=req.instruction,
            test_cases=req.test_cases,
            max_retries=max(1, min(req.max_retries, 5)),
            task_mode=req.task_mode,
            session_id=req.session_id,
        )
        return {
            "status": "ok",
            "all_passed": result["all_passed"],
            "attempts": result["attempts"],
            "test_results": result["test_results"],
            "diff": result["diff"],
            "file_path": resolved_path,
            "explanation": result.get("explanation", ""),
            "citations": result.get("citations", []),
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}


@app.post("/clear")
async def clear():
    session_chat_history.clear()
    return {"status": "cleared"}


@app.post("/clear_memory")
async def clear_memory():
    """Clear code agent session memory."""
    from agent.code_agent import get_session_memory
    memory = get_session_memory()
    memory.clear_all()
    return {"status": "memory_cleared"}


# ── Persistent chat session endpoints ────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    title: str = "New Chat"

class RenameSessionRequest(BaseModel):
    session_id: str
    title: str

class DeleteSessionRequest(BaseModel):
    session_id: str

class GetSessionHistoryRequest(BaseModel):
    session_id: str


@app.get("/sessions")
async def list_sessions():
    """List all chat sessions."""
    return {"sessions": chat_store.list_sessions()}


@app.post("/sessions/create")
async def create_session(req: CreateSessionRequest):
    """Create a new named chat session."""
    session_id = chat_store.create_session(req.title)
    return {"session_id": session_id, "title": req.title}


@app.post("/sessions/rename")
async def rename_session(req: RenameSessionRequest):
    ok = chat_store.rename_session(req.session_id, req.title)
    return {"status": "ok" if ok else "error"}


@app.post("/sessions/delete")
async def delete_session(req: DeleteSessionRequest):
    count = chat_store.delete_session(req.session_id)
    return {"status": "ok", "deleted": count}


@app.post("/sessions/history")
async def get_session_history(req: GetSessionHistoryRequest):
    """Get full conversation history for a session."""
    turns = chat_store.get_turns(req.session_id)
    return {"turns": turns}
