import asyncio
import warnings
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Literal, Optional
from pathlib import Path

from brain.fast_search import initialize_bm25
from brain.ingest import ingest_docs
from brain.ingest import ingest_file
from brain.ingest import _reingest_log
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
    docs, topic_synonyms = await asyncio.to_thread(ingest_docs)
    raw_docs = load_pdfs(DATA_DIR)
    return {"topics": list(topic_synonyms.keys())}


# ── Reingest: background task with status polling ────────────────────────────
_reingest_state: dict = {"status": "idle", "topics": [], "error": "", "log": [], "log_cursor": 0}

async def _run_reingest():
    global raw_docs, _reingest_state
    _reingest_state = {"status": "running", "topics": [], "error": "", "log": [], "log_cursor": 0}
    try:
        print("[API] reingest background task started")
        docs, topic_synonyms = await asyncio.to_thread(ingest_docs, force=True)
        raw_docs = load_pdfs(DATA_DIR)
        initialize_bm25(raw_docs)
        _reingest_state["status"] = "done"
        _reingest_state["topics"] = list(topic_synonyms.keys())
        _reingest_state["log"] = list(_reingest_log)
        print(f"[API] reingest complete: {len(_reingest_state['topics'])} topics")
    except Exception as e:
        import traceback
        _reingest_state["status"] = "error"
        _reingest_state["error"] = str(e)
        _reingest_state["log"] = list(_reingest_log)
        print(f"[API] reingest failed: {e}\n{traceback.format_exc()}")


@app.post("/reingest")
async def reingest():
    """Start a full re-ingestion in the background. Poll /reingest/status for progress."""
    global _reingest_state
    if _reingest_state["status"] == "running":
        return {"status": "already_running"}
    _reingest_state = {"status": "running", "topics": [], "error": "", "log": [], "log_cursor": 0}
    asyncio.create_task(_run_reingest())
    return {"status": "started"}


@app.get("/reingest/status")
async def reingest_status(log_cursor: int = 0):
    """Poll this for progress. Pass log_cursor=N to get only new log lines since N."""
    full_log = list(_reingest_log) if _reingest_state["status"] == "running" else _reingest_state.get("log", [])
    new_lines = full_log[log_cursor:]
    return {
        "status": _reingest_state["status"],
        "topics": _reingest_state.get("topics", []),
        "error": _reingest_state.get("error", ""),
        "log": new_lines,
        "log_cursor": len(full_log),
    }


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

        result = await asyncio.to_thread(ingest_file, str(dest))
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
        result = await asyncio.to_thread(ingest_file, filename)
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
        
        # Build the full assistant response matching what the frontend displays
        response_parts = []
        if dry_run_output and dry_run_output != "(No changes — agent could not determine what to modify.)":
            response_parts.append(f"[DRY RUN PREVIEW]\n\n{dry_run_output}")
        if explanation:
            citation_block = ""
            if citations:
                citation_block = "\n\n📚 Sources:\n" + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(citations))
            response_parts.append(f"[EXPLANATION]\n{explanation}{citation_block}")
        full_response = "\n\n".join(response_parts) if response_parts else explanation

        # Store agent interaction into the shared query chat history
        # so follow-up questions in query mode have context
        session_chat_history.append({"role": "User", "content": req.instruction})
        if full_response:
            session_chat_history.append({"role": "Assistant", "content": full_response})

        # Also persist to chat session store for cross-restart memory
        if req.session_id:
            chat_store.add_turn(req.session_id, "User", f"[AGENT] {req.instruction}")
            if full_response:
                chat_store.add_turn(req.session_id, "Assistant", full_response)
        
        # Extract new_source for the Apply button to write directly
        new_source = None
        if isinstance(result, dict):
            new_source = result.get("new_source")

        return {
            "status": "pending_review",
            "dry_run_output": dry_run_output,
            "file_path": resolved_path,
            "explanation": explanation,
            "citations": citations,
            "new_source": new_source,
        }
    except Exception as e:
        return {"error": str(e), "status": "error", "message": f"Agent error: {str(e)}"}


class AgentApplyRequest(BaseModel):
    instruction: str
    file_path: str
    confirmed: bool = False
    task_mode: str = "auto"  # "fix", "solve", or "auto"
    session_id: Optional[str] = None
    new_source: Optional[str] = None  # pre-computed source to write directly
    context_file_edits: Optional[list] = None  # [{path, new_source}] for multi-file


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
        # If new_source was provided (from dry_run preview), write it directly
        # instead of re-running the LLM (which would be non-deterministic).
        if req.new_source is not None:
            import os, subprocess
            with open(resolved_path, "w", encoding="utf-8") as f:
                f.write(req.new_source)
            # Run gofmt for Go files
            ext = os.path.splitext(resolved_path)[1].lower()
            if ext == ".go":
                try:
                    subprocess.run(["gofmt", "-w", resolved_path], timeout=5, check=True)
                except Exception:
                    pass

            # Also apply context file edits if provided
            applied_context = []
            if req.context_file_edits:
                for cf_edit in req.context_file_edits:
                    cf_path = cf_edit.get("path", "")
                    cf_source = cf_edit.get("new_source", "")
                    if cf_path and cf_source and os.path.isfile(cf_path):
                        with open(cf_path, "w", encoding="utf-8") as cf:
                            cf.write(cf_source)
                        cf_ext = os.path.splitext(cf_path)[1].lower()
                        if cf_ext == ".go":
                            try:
                                subprocess.run(["gofmt", "-w", cf_path], timeout=5, check=True)
                            except Exception:
                                pass
                        applied_context.append(cf_path)

            return {
                "status": "success",
                "message": f"Changes applied to {resolved_path}",
                "applied_context_files": applied_context,
            }

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

        new_source = None
        if isinstance(result, dict):
            new_source = result.get("new_source")

        return {
            "status": "pending_review",
            "dry_run_output": dry_run_output,
            "file_path": resolved_path,
            "max_chunks": max_chunks,
            "new_source": new_source,
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
    test_cases: list = []          # optional [{input: str, expected: str}] for execution-based critique


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
            test_cases=req.test_cases if req.test_cases else None,
        )

        # Build the full assistant response matching what the frontend displays
        response_parts = []
        plan = result.get("plan", {})
        plan_steps = plan.get("steps", [])
        if plan_steps:
            formatted_steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan_steps))
            response_parts.append(f"[PLAN]\n{formatted_steps}")
        diff = result.get("diff", "")
        if diff:
            response_parts.append(f"[DRY RUN PREVIEW]\n\n{diff}")
        explanation = result.get("explanation", "")
        if explanation:
            citations = result.get("citations", [])
            related_files = result.get("related_files", [])
            citation_block = ""
            if citations:
                citation_block = "\n\n📚 Sources:\n" + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(citations))
            related_note = ""
            if related_files:
                related_note = f"\n📎 Related files read: {', '.join(str(f) for f in related_files)}"
            response_parts.append(f"[EXPLANATION]\n{explanation}{citation_block}{related_note}")
        verdict = result.get("verdict", "")
        attempts = result.get("attempts", 0)
        if verdict:
            verdict_text = f"[VERDICT] {verdict} — {attempts} attempt(s)"
            critic_feedback = result.get("critic_feedback", "")
            if verdict == "FAIL" and critic_feedback:
                verdict_text += f"\n\n[CRITIC FEEDBACK]\n{critic_feedback}"
            response_parts.append(verdict_text)
        full_response = "\n\n".join(response_parts) if response_parts else explanation

        # Persist agent conversation to chat session store
        if req.session_id:
            chat_store.add_turn(req.session_id, "User", f"[AGENT] {req.instruction}")
            if full_response:
                chat_store.add_turn(req.session_id, "Assistant", full_response)
        # Also push to in-memory history for query follow-ups
        session_chat_history.append({"role": "User", "content": req.instruction})
        if full_response:
            session_chat_history.append({"role": "Assistant", "content": full_response})

        # Always return the full multi-agent result; frontend uses `status`
        # field ("pending_review" when changes exist) to show approval buttons.
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

    # Save original file content so we can restore it after the pipeline.
    # fix_with_tests writes intermediate states to disk; the user must
    # approve changes via the frontend before they are persisted.
    try:
        original_content = Path(resolved_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        original_content = None

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

        # Restore original file on disk — user must approve via frontend
        if original_content is not None:
            try:
                Path(resolved_path).write_text(original_content, encoding="utf-8")
            except Exception:
                pass

        return {
            "status": "ok",
            "all_passed": result["all_passed"],
            "attempts": result["attempts"],
            "test_results": result["test_results"],
            "diff": result["diff"],
            "file_path": resolved_path,
            "explanation": result.get("explanation", ""),
            "citations": result.get("citations", []),
            "new_source": result.get("final_source", ""),
        }
    except Exception as e:
        # Restore on error too
        if original_content is not None:
            try:
                Path(resolved_path).write_text(original_content, encoding="utf-8")
            except Exception:
                pass
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
    offset: int = 0    # pagination: skip first N turns
    limit: int = 0     # 0 = all turns (default for backward compat)


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
    """Get conversation history for a session with optional pagination."""
    turns = chat_store.get_turns(req.session_id)
    total = len(turns)
    if req.limit > 0:
        turns = turns[req.offset:req.offset + req.limit]
    elif req.offset > 0:
        turns = turns[req.offset:]
    return {"turns": turns, "total": total, "offset": req.offset}


# ── Voice I/O endpoints ─────────────────────────────────────────────────────

@app.post("/voice/transcribe")
async def voice_transcribe(audio: UploadFile = File(...)):
    """Transcribe an audio file to text via Whisper."""
    try:
        from agent.voice_io import transcribe_file
        import tempfile, os
        # Frontend always sends WAV now (Web Audio API PCM encoder — no ffmpeg needed)
        suffix = os.path.splitext(audio.filename or ".wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
        try:
            result = transcribe_file(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        # transcribe_file returns {"text": str, "language": str, "segments": list}
        text = result["text"] if isinstance(result, dict) else str(result)
        return {"status": "ok", "text": text}
    except Exception as e:
        return {"status": "error", "error": str(e)}


class TTSRequest(BaseModel):
    text: str

@app.post("/voice/speak")
async def voice_speak(req: TTSRequest):
    """Convert text to speech (plays on server or returns audio path)."""
    try:
        from agent.voice_io import speak_text
        speak_text(req.text)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── OCR endpoints ────────────────────────────────────────────────────────────

@app.post("/ocr/extract")
async def ocr_extract(
    image: UploadFile = File(...),
    mode: str = "auto",
):
    """Extract text from a screenshot image.

    mode: 'auto', 'code', 'diagram', 'text'
    """
    try:
        from agent.ocr_capture import extract_text_from_image
        image_bytes = await image.read()
        result = extract_text_from_image(image_bytes, mode=mode)
        return {"status": "ok", **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/ocr/analyze")
async def ocr_analyze(
    image: UploadFile = File(...),
    question: str = Form(""),
    mode: str = Form("auto"),
):
    """Analyze a screenshot image using it as visual evidence for the LLM.

    Instead of just extracting text, the image is used as supporting context
    so the LLM can reason about what it shows alongside the user's question.
    """
    try:
        from agent.ocr_capture import analyze_image_with_context
        image_bytes = await image.read()
        result = analyze_image_with_context(image_bytes, user_question=question, mode=mode)
        return {"status": "ok", **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Tool Hooks: Linting, Testing, Logs, Git ──────────────────────────────────

class LintRequest(BaseModel):
    file_path: str
    fix: bool = False

class PytestRequest(BaseModel):
    target: str = "."
    with_coverage: bool = False
    cwd: Optional[str] = None

class LogParseRequest(BaseModel):
    log_path: str
    tail: int = 500

class GitDiffRequest(BaseModel):
    ref: str = "HEAD"
    file_path: Optional[str] = None

class GitLogRequest(BaseModel):
    count: int = 20

class GitBlameRequest(BaseModel):
    file_path: str
    line_number: int


@app.post("/tools/lint")
async def lint_endpoint(req: LintRequest):
    """Run language-appropriate linter on a file."""
    try:
        from agent.tool_hooks import lint_file
        result = lint_file(req.file_path, fix=req.fix)
        return {"status": "ok", **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/tools/type_check")
async def type_check_endpoint(req: LintRequest):
    """Run type checker (mypy for Python, tsc for TypeScript)."""
    try:
        from agent.tool_hooks import run_mypy, run_tsc_check, _detect_language
        lang = _detect_language(req.file_path)
        if lang == "python":
            result = run_mypy(req.file_path)
        elif lang in ("typescript", "javascript"):
            result = run_tsc_check(req.file_path)
        else:
            return {"status": "ok", "tool": "none", "message": f"No type checker for {lang}"}
        return {"status": "ok", **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/tools/pytest")
async def pytest_endpoint(req: PytestRequest):
    """Run pytest with optional coverage."""
    try:
        from agent.tool_hooks import run_pytest, run_pytest_with_coverage
        if req.with_coverage:
            result = run_pytest_with_coverage(req.target, cwd=req.cwd)
        else:
            result = run_pytest(req.target, cwd=req.cwd)
        return {"status": "ok", **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/tools/parse_logs")
async def parse_logs_endpoint(req: LogParseRequest):
    """Parse a log file for errors and group repeated patterns."""
    try:
        from agent.tool_hooks import read_log_file
        result = read_log_file(req.log_path, tail=req.tail)
        return result
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/tools/git_diff")
async def git_diff_endpoint(req: GitDiffRequest):
    """Get git diff for code review."""
    try:
        from agent.tool_hooks import git_diff
        result = git_diff(ref=req.ref, file_path=req.file_path)
        return {"status": "ok", **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/tools/git_diff_staged")
async def git_diff_staged_endpoint():
    """Get staged changes diff."""
    try:
        from agent.tool_hooks import git_diff_staged
        result = git_diff_staged()
        return {"status": "ok", **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/tools/git_log")
async def git_log_endpoint(req: GitLogRequest):
    """Get readable changelog from recent commits."""
    try:
        from agent.tool_hooks import git_log_summary
        result = git_log_summary(count=req.count)
        return {"status": "ok", **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/tools/git_blame")
async def git_blame_endpoint(req: GitBlameRequest):
    """Get blame info for a specific line."""
    try:
        from agent.tool_hooks import git_blame_line
        result = git_blame_line(req.file_path, req.line_number)
        return {"status": "ok", **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/tools/pre_commit")
async def pre_commit_endpoint():
    """Run pre-commit style checks on staged files."""
    try:
        from agent.tool_hooks import pre_commit_check
        result = pre_commit_check()
        return {"status": "ok", **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Tutor Mode endpoints ────────────────────────────────────────────────────

class TutorStartRequest(BaseModel):
    topic: str
    difficulty: str = "medium"  # "easy", "medium", "hard"
    language: str = "python"
    style: str = "mcq"  # "mcq", "free_text", "code"

class TutorAnswerRequest(BaseModel):
    session_id: str
    answer: str

class TutorCodeRequest(BaseModel):
    session_id: str
    code: str

class TutorHintRequest(BaseModel):
    session_id: str


@app.post("/tutor/start")
async def tutor_start(req: TutorStartRequest):
    """Generate a new tutor problem."""
    from agent.tutor import generate_problem
    result = generate_problem(
        topic=req.topic,
        difficulty=req.difficulty,
        language=req.language,
        style=req.style,
    )
    return {"status": "ok", **result}


@app.post("/tutor/check")
async def tutor_check(req: TutorAnswerRequest):
    """Check the user's answer (MCQ letter or free text)."""
    from agent.tutor import check_answer
    result = check_answer(req.session_id, req.answer)
    return {"status": "ok", **result}


@app.post("/tutor/run")
async def tutor_run(req: TutorCodeRequest):
    """Run user code against the problem's test cases."""
    from agent.tutor import run_tutor_code
    result = run_tutor_code(req.session_id, req.code)
    return {"status": "ok", **result}


@app.post("/tutor/hint")
async def tutor_hint(req: TutorHintRequest):
    """Get the next progressive hint."""
    from agent.tutor import get_hint
    result = get_hint(req.session_id)
    return {"status": "ok", **result}


@app.get("/tutor/learnings")
async def tutor_learnings(topic: str = "", language: str = "", limit: int = 5):
    """Get recent code agent learnings for use in tutor mode."""
    from agent.tutor import get_agent_learnings
    learnings = get_agent_learnings(topic=topic, language=language, limit=limit)
    return {"status": "ok", "learnings": learnings}
