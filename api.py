import asyncio
from datetime import datetime
import logging
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import warnings
from collections import deque
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
from brain.config import DATA_DIR, OUTPUT_DIR


warnings.filterwarnings("ignore")

from print_logger import get_logger
_api_log = get_logger("tutor")  # reuse the tutor logger so messages flow to SSE

raw_docs = []
chat_store = ChatSessionStore()

# ── Cancellation flag for long-running agent tasks ────────────────────────────
import threading
_cancel_event = threading.Event()  # set() to cancel, clear() to reset


def check_cancelled():
    """Check if a cancellation has been requested. Agent code can call this."""
    return _cancel_event.is_set()

# Register the cancellation check with TimedLLM so every LLM call can bail out
from brain.timed_llm import register_cancel_check
register_cancel_check(check_cancelled)

# ── Process log streaming (SSE) ───────────────────────────────────────────────
# Captures logs from agent/brain modules and streams them to the frontend
# in real-time via Server-Sent Events so the user can see backend thinking.

_LOG_SUBSCRIBERS: list[asyncio.Queue] = []
_LOG_HISTORY: deque = deque(maxlen=200)  # keep last 200 log entries for late joiners


class _SSELogHandler(logging.Handler):
    """Logging handler that pushes records into all active SSE subscriber queues."""

    def emit(self, record: logging.LogRecord):
        entry = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "message": self.format(record),
        }
        _LOG_HISTORY.append(entry)
        for q in _LOG_SUBSCRIBERS:
            try:
                q.put_nowait(entry)
            except asyncio.QueueFull:
                pass  # slow consumer — drop oldest


def _setup_log_streaming():
    """Attach SSE handler to all agent/brain loggers."""
    handler = _SSELogHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    for name in ("code_agent", "multi_agent", "tool_hooks", "ocr", "voice_io", "chat_session_store", "tutor", "fast_search", "rag_brain", "query_pipeline", "code_context", "llm_calls"):
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        lg.setLevel(logging.DEBUG)


_API_PORT = int(os.getenv("AION_API_PORT", "8000"))


def _kill_stale_port(port: int = _API_PORT):
    """Kill any leftover process holding our API port (Windows-only).
    
    This prevents 'address already in use' errors from dead uvicorn workers
    that survived after the Electron app or user closed without clean shutdown.
    """
    if sys.platform != "win32":
        return
    try:
        # netstat -ano finds PIDs bound to the port
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=5,
        )
        my_pid = os.getpid()
        killed = set()
        for line in result.stdout.splitlines():
            # Match lines like:  TCP  0.0.0.0:8000  ...  LISTENING  12345
            if f":{port} " not in line and f":{port}\t" not in line:
                continue
            if "LISTENING" not in line:
                continue
            parts = line.split()
            pid_str = parts[-1]
            if not pid_str.isdigit():
                continue
            pid = int(pid_str)
            if pid == my_pid or pid in killed or pid == 0:
                continue
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True, timeout=5,
                )
                killed.add(pid)
                print(f"[API] Killed stale process PID {pid} on port {port}")
            except Exception:
                pass
        if killed:
            time.sleep(0.5)  # brief pause for OS to release the port
    except Exception as e:
        print(f"[API] Port cleanup warning: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global raw_docs
    _kill_stale_port()
    _setup_log_streaming()
    raw_docs = load_pdfs(DATA_DIR)
    initialize_bm25(raw_docs)
    yield
    # Shutdown: nothing to clean up — uvicorn handles socket close

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


from starlette.requests import Request
from starlette.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print(f"[UNHANDLED ERROR] {request.url.path}: {exc}\n{tb}", flush=True)
    _api_log.error("[UNHANDLED ERROR] %s: %s\n%s", request.url.path, exc, tb)
    return JSONResponse(status_code=500, content={"status": "error", "error": str(exc)})


@app.get("/health")
async def health_check():
    """Check if the backend is up and the LLM (Ollama) is reachable."""
    from brain.config import LLM_MODEL
    import httpx
    llm_ok = False
    llm_model = LLM_MODEL
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                available = [m.get("name", "") for m in data.get("models", [])]
                # Check if our configured model is available (handle tag suffixes)
                llm_ok = any(LLM_MODEL in name for name in available)
    except Exception:
        pass
    return {"status": "ok", "llm_connected": llm_ok, "llm_model": llm_model}


# ── Model management endpoints ────────────────────────────────────────────────

@app.get("/models")
async def list_models():
    """List all available Ollama models and the currently active one."""
    import brain.config as cfg
    import httpx
    available = []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("models", []):
                    available.append({
                        "name": m.get("name", ""),
                        "size": m.get("size", 0),
                        "modified_at": m.get("modified_at", ""),
                    })
    except Exception as e:
        return {"status": "error", "error": f"Cannot reach Ollama: {e}"}
    return {
        "status": "ok",
        "current_model": cfg.LLM_MODEL,
        "available": available,
    }


class SwitchModelRequest(BaseModel):
    model: str


@app.post("/models/switch")
async def switch_model(req: SwitchModelRequest):
    """Switch the active LLM model at runtime. Validates the model exists in Ollama first."""
    import brain.config as cfg
    import httpx

    new_model = req.model.strip()
    if not new_model:
        return {"status": "error", "error": "Model name cannot be empty"}

    # Verify model exists in Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                available = [m.get("name", "") for m in data.get("models", [])]
                # Match with or without tag suffix (e.g. "qwen3.5" matches "qwen3.5:latest")
                found = any(new_model in name or name.split(":")[0] == new_model for name in available)
                if not found:
                    return {
                        "status": "error",
                        "error": f"Model '{new_model}' not found in Ollama",
                        "available": available,
                    }
    except Exception as e:
        return {"status": "error", "error": f"Cannot reach Ollama: {e}"}

    old_model = cfg.LLM_MODEL
    cfg.LLM_MODEL = new_model
    return {
        "status": "ok",
        "previous_model": old_model,
        "current_model": new_model,
    }


# ── Process log SSE stream ────────────────────────────────────────────────────
@app.get("/process/logs")
async def process_logs_stream():
    """SSE stream of backend process logs (agent thinking, RAG search, etc.)."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    _LOG_SUBSCRIBERS.append(queue)

    async def event_generator():
        try:
            # Send recent history so late joiners see context
            for entry in list(_LOG_HISTORY)[-50:]:
                yield f"data: {json.dumps(entry)}\n\n"
            # Stream new logs as they arrive
            while True:
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(entry)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"  # prevent connection timeout
        except asyncio.CancelledError:
            pass
        finally:
            _LOG_SUBSCRIBERS.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/process/logs/history")
async def process_logs_history():
    """Return recent log history (last 200 entries) for initial load."""
    return {"logs": list(_LOG_HISTORY)}


# ── Cancel all running agent processes ─────────────────────────────────────────
@app.post("/agent/cancel")
async def agent_cancel():
    """Signal all running agent/LLM tasks to stop ASAP."""
    _cancel_event.set()
    _api_log.info("[API] Cancellation requested — all running agent tasks will stop")
    return {"status": "ok", "message": "Cancellation signal sent"}


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
    # Always clear any stale cancellation from a previous agent operation so
    # the chat endpoint is never silently broken after a user cancels the agent.
    _cancel_event.clear()
    try:
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
    except Exception as e:
        _api_log.error("query failed: %s\n%s", e, traceback.format_exc())
        return {"answer": f"Error: {e}", "error": str(e), "status": "error"}

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
    try:
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
    except Exception as e:
        _api_log.error("feedback failed: %s\n%s", e, traceback.format_exc())
        return {"answer": f"Error: {e}", "error": str(e), "status": "error"}


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
            try:
                matches = list(Path(".").rglob(candidate.name))
            except OSError as e:
                _api_log.warning("rglob scan failed (%s) — cannot locate %s", e, candidate.name)
                matches = []
            if matches:
                resolved_path = str(matches[0].resolve())
            else:
                return {"error": f"File not found: {req.file_path}", "status": "error",
                        "message": f"File not found: {req.file_path}. Please provide the full absolute path."}
        else:
            return {"error": f"File not found: {req.file_path}", "status": "error",
                    "message": f"File not found: {req.file_path}"}

    def _run_edit():
        _cancel_event.clear()
        from agent.agent_metrics import MetricsCollector, persist_summary
        with MetricsCollector("edit_code") as mc:
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
            persist_summary(mc.summary())
            return result

    try:
        result = await asyncio.to_thread(_run_edit)

        if _cancel_event.is_set():
            _cancel_event.clear()
            return {"status": "error", "error": "Cancelled by user"}

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
        from brain.timed_llm import CancelledError
        if isinstance(e, CancelledError) or _cancel_event.is_set():
            _cancel_event.clear()
            return {"status": "error", "error": "Cancelled by user"}
        _api_log.error("agent_edit failed: %s\n%s", e, traceback.format_exc())
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
        def _apply_edit():
            from agent.agent_metrics import MetricsCollector, persist_summary
            with MetricsCollector("apply_edit") as mc:
                result = agent.edit_code(
                    path=resolved_path,
                    instruction=req.instruction,
                    dry_run=False,
                    use_rag=True,
                    rerank_method="cross_encoder",  # always use best quality when actually writing
                    task_mode=req.task_mode,
                    session_id=req.session_id,
                )
                persist_summary(mc.summary())
                return result
        result = await asyncio.to_thread(_apply_edit)
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

        def _run_chunks():
            _cancel_event.clear()
            from agent.agent_metrics import MetricsCollector, persist_summary
            with MetricsCollector("edit_with_chunks") as mc:
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
                persist_summary(mc.summary())
                return result
        result = await asyncio.to_thread(_run_chunks)

        if _cancel_event.is_set():
            _cancel_event.clear()
            return {"status": "error", "error": "Cancelled by user"}

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
        from brain.timed_llm import CancelledError
        if isinstance(e, CancelledError) or _cancel_event.is_set():
            _cancel_event.clear()
            return {"status": "error", "error": "Cancelled by user"}
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
        def _run_orchestrate():
            _cancel_event.clear()
            from agent.agent_metrics import MetricsCollector, persist_summary
            with MetricsCollector("orchestrate") as mc:
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
                persist_summary(mc.summary())
                return result
        result = await asyncio.to_thread(_run_orchestrate)

        if _cancel_event.is_set():
            _cancel_event.clear()
            return {"status": "error", "error": "Cancelled by user"}

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
        from brain.timed_llm import CancelledError
        if isinstance(e, CancelledError) or _cancel_event.is_set():
            _cancel_event.clear()
            return {"status": "error", "error": "Cancelled by user"}
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
        def _run():
            source = agent_read_file(resolved_path)
            return run_tests(source, resolved_path, req.test_cases)
        results = await asyncio.to_thread(_run)
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
        def _run_fix():
            _cancel_event.clear()
            from agent.agent_metrics import MetricsCollector, persist_summary
            with MetricsCollector("fix_with_tests") as mc:
                agent = CodeAgent(repo_path=".")
                result = agent.fix_with_tests(
                    path=resolved_path,
                    instruction=req.instruction,
                    test_cases=req.test_cases,
                    max_retries=max(1, min(req.max_retries, 5)),
                    task_mode=req.task_mode,
                    session_id=req.session_id,
                )
                persist_summary(mc.summary())
                return result
        result = await asyncio.to_thread(_run_fix)

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
        from brain.timed_llm import CancelledError
        if isinstance(e, CancelledError) or _cancel_event.is_set():
            _cancel_event.clear()
            return {"status": "error", "error": "Cancelled by user"}
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
    """Generate a new tutor problem. Auto-detects math topics and routes to math tutor."""
    from tutor.tutor import generate_problem, generate_math_problem, is_math_topic
    _is_math = is_math_topic(req.topic)
    _api_log.info("[API] /tutor/start — topic=%s, math=%s, style=%s", req.topic, _is_math, req.style)
    def _gen():
        if _is_math:
            math_style = "mcq" if req.style == "mcq" else "solve"
            return generate_math_problem(topic=req.topic, difficulty=req.difficulty, style=math_style)
        return generate_problem(topic=req.topic, difficulty=req.difficulty, language=req.language, style=req.style)
    result = await asyncio.to_thread(_gen)
    _api_log.info("[API] /tutor/start — done ✓")
    return {"status": "ok", **result}


@app.post("/tutor/check")
async def tutor_check(req: TutorAnswerRequest):
    """Check the user's answer (MCQ letter, free text, or math answer)."""
    _api_log.info("[API] /tutor/check — session=%s", req.session_id)
    from tutor.tutor import check_answer, check_math_answer, _tutor_sessions
    state = _tutor_sessions.get(req.session_id)
    def _check():
        if state and state.get("is_math"):
            return check_math_answer(req.session_id, req.answer)
        return check_answer(req.session_id, req.answer)
    result = await asyncio.to_thread(_check)
    return {"status": "ok", **result}


@app.post("/tutor/run")
async def tutor_run(req: TutorCodeRequest):
    """Run user code against the problem's test cases."""
    from tutor.tutor import run_tutor_code
    result = run_tutor_code(req.session_id, req.code)
    return {"status": "ok", **result}


@app.post("/tutor/hint")
async def tutor_hint(req: TutorHintRequest):
    """Get the next progressive hint."""
    from tutor.tutor import get_hint
    result = await asyncio.to_thread(get_hint, req.session_id)
    return {"status": "ok", **result}


@app.get("/tutor/learnings")
async def tutor_learnings(topic: str = "", language: str = "", limit: int = 5):
    """Get recent code agent learnings for use in tutor mode."""
    from tutor.tutor import get_agent_learnings
    learnings = get_agent_learnings(topic=topic, language=language, limit=limit)
    return {"status": "ok", "learnings": learnings}


# ── Math Tutor endpoints ───────────────────────────────────────────────────

class MathStartRequest(BaseModel):
    topic: str
    difficulty: str = "medium"
    style: str = "solve"  # "solve", "mcq", "proof"

class MathAnswerRequest(BaseModel):
    session_id: str
    answer: str

class MathEvalRequest(BaseModel):
    expression: str
    variable: str = "x"
    values: Optional[list] = None  # custom x-values; defaults to -10..10

class MatrixRequest(BaseModel):
    operation: str  # "add", "subtract", "multiply", "determinant", "inverse", "transpose"
    matrices: list  # list of 2D arrays


@app.post("/math/start")
async def math_start(req: MathStartRequest):
    """Generate a new math practice problem with lesson and step-by-step solution."""
    _api_log.info("[API] /math/start — topic=%s, difficulty=%s, style=%s", req.topic, req.difficulty, req.style)
    from tutor.tutor import generate_math_problem
    result = await asyncio.to_thread(generate_math_problem, topic=req.topic, difficulty=req.difficulty, style=req.style)
    _api_log.info("[API] /math/start — done ✓")
    return {"status": "ok", **result}


@app.post("/math/check")
async def math_check(req: MathAnswerRequest):
    """Check a math answer (supports equivalent forms via LLM evaluation)."""
    _api_log.info("[API] /math/check — session=%s", req.session_id)
    from tutor.tutor import check_math_answer
    result = await asyncio.to_thread(check_math_answer, req.session_id, req.answer)
    _api_log.info("[API] /math/check — done ✓")
    return {"status": "ok", **result}


@app.post("/math/hint")
async def math_hint(req: TutorHintRequest):
    """Get the next progressive hint for a math problem."""
    _api_log.info("[API] /math/hint — session=%s", req.session_id)
    from tutor.tutor import get_hint
    result = await asyncio.to_thread(get_hint, req.session_id)
    _api_log.info("[API] /math/hint — done ✓")
    return {"status": "ok", **result}


@app.get("/math/steps/{session_id}")
async def math_steps(session_id: str):
    """Get the full step-by-step solution (available after solving or 3 attempts)."""
    _api_log.info("[API] /math/steps — session=%s", session_id)
    from tutor.tutor import get_math_step_by_step
    result = get_math_step_by_step(session_id)
    _api_log.info("[API] /math/steps — done ✓")
    return {"status": "ok", **result}


@app.post("/math/evaluate")
async def math_evaluate(req: MathEvalRequest):
    """Evaluate a math expression at given points for interactive graphing."""
    _api_log.info("[API] /math/evaluate — expr=%s", req.expression[:40])
    from tutor.tutor import evaluate_math_expression
    result = evaluate_math_expression(
        expression=req.expression,
        variable=req.variable,
        values=req.values,
    )
    _api_log.info("[API] /math/evaluate — done ✓")
    return {"status": "ok", **result}


@app.post("/math/matrix")
async def math_matrix(req: MatrixRequest):
    """Solve a matrix operation with step-by-step solution."""
    from tutor.tutor import solve_matrix_problem
    result = solve_matrix_problem(operation=req.operation, matrices=req.matrices)
    return {"status": "ok", **result}


# ── Curriculum endpoints ──────────────────────────────────────────────────

@app.get("/math/curriculum")
async def math_curriculum():
    """Get the full curriculum tree with progress."""
    from tutor.tutor import get_curriculum
    return {"status": "ok", "curriculum": get_curriculum()}


@app.get("/math/curriculum/{subject_id}/{chapter_id}")
async def math_chapter(subject_id: str, chapter_id: str):
    """Get topics for a specific chapter."""
    from tutor.tutor import get_chapter_topics
    result = get_chapter_topics(subject_id, chapter_id)
    return {"status": "ok", **result}


class ChapterProgressRequest(BaseModel):
    subject_id: str
    chapter_id: str

@app.post("/math/curriculum/progress")
async def math_progress(req: ChapterProgressRequest):
    """Mark a chapter as completed."""
    from tutor.tutor import mark_chapter_complete
    result = mark_chapter_complete(req.subject_id, req.chapter_id)
    return result


# ── CS Curriculum endpoints ───────────────────────────────────────────────

@app.get("/cs/curriculum")
async def cs_curriculum():
    """Get the full CS curriculum tree with progress."""
    from tutor.tutor import get_cs_curriculum
    return {"status": "ok", "curriculum": get_cs_curriculum()}


@app.get("/cs/curriculum/{subject_id}/{chapter_id}")
async def cs_chapter(subject_id: str, chapter_id: str):
    """Get topics for a specific CS chapter."""
    from tutor.tutor import get_cs_chapter_topics
    result = get_cs_chapter_topics(subject_id, chapter_id)
    return {"status": "ok", **result}


class CSChapterProgressRequest(BaseModel):
    subject_id: str
    chapter_id: str

@app.post("/cs/curriculum/progress")
async def cs_progress(req: CSChapterProgressRequest):
    """Mark a CS chapter as completed."""
    from tutor.tutor import mark_cs_chapter_complete
    result = mark_cs_chapter_complete(req.subject_id, req.chapter_id)
    return result


# ── Gamification endpoints ────────────────────────────────────────────────

@app.get("/gamification/profile")
async def gamification_profile():
    """Get the user's XP, level, badges, streak, and stats."""
    from tutor.gamification import get_profile
    return {"status": "ok", **get_profile()}


@app.post("/gamification/reset")
async def gamification_reset():
    """Reset the gamification profile (fresh start)."""
    from tutor.gamification import reset_profile
    reset_profile()
    return {"status": "ok", "message": "Profile reset"}


# ── Agent metrics endpoints ──────────────────────────────────────────────

@app.get("/agent/metrics")
async def agent_metrics():
    """Get recent agent operation metrics and aggregate stats."""
    from agent.agent_metrics import load_history, aggregate_stats
    history = load_history(last_n=50)
    return {
        "status": "ok",
        "recent": history,
        "aggregate": aggregate_stats(history),
    }


# ── Visualization modules endpoints ──────────────────────────────────────

@app.get("/visualizations")
async def visualizations_list():
    """Get all available interactive math visualization modules."""
    from tutor.math_visualizations import get_visualization_modules
    return {"status": "ok", "modules": get_visualization_modules()}


@app.get("/visualizations/{module_id}")
async def visualization_detail(module_id: str):
    """Get details for a specific visualization module."""
    from tutor.math_visualizations import get_module
    mod = get_module(module_id)
    if not mod:
        return {"status": "error", "message": f"Module '{module_id}' not found"}
    return {"status": "ok", "module": mod}


# ── Problem bank endpoints ───────────────────────────────────────────────

@app.get("/problem-bank/stats")
async def problem_bank_stats():
    """Get stats about the verified problem bank."""
    from tutor.problem_bank import get_bank_stats
    return {"status": "ok", **get_bank_stats()}


@app.post("/problem-bank/ingest")
async def problem_bank_ingest():
    """Ingest verified problems from RAG documents into the problem bank."""
    from tutor.problem_bank import ingest_problems_from_rag
    result = await asyncio.to_thread(ingest_problems_from_rag)
    return {"status": "ok", **result}


class ProblemBankGetRequest(BaseModel):
    category: str = "general"
    difficulty: str = "medium"

@app.post("/problem-bank/get")
async def problem_bank_get(req: ProblemBankGetRequest):
    """Get a verified problem from the bank."""
    from tutor.problem_bank import get_problem
    problem = get_problem(category=req.category, difficulty=req.difficulty)
    if not problem:
        return {"status": "error", "message": "No problems available for that category/difficulty"}
    return {"status": "ok", "problem": problem}


# ── Two-mode agent endpoint ─────────────────────────────────────────────

class TwoModeRequest(BaseModel):
    instruction: str
    source_code: str = ""
    file_path: str = "solution.py"
    test_cases: list = []

@app.post("/agent/two-mode")
async def agent_two_mode(req: TwoModeRequest):
    """Run the two-mode code agent (auto-routes between Do-It and Explain)."""
    _api_log.info("[API] /agent/two-mode — instruction=%s", req.instruction[:60])
    from agent.multi_agent_graph import run_two_mode_agent

    # Resolve file_path the same way other agent endpoints do
    resolved_path = req.file_path
    if not Path(resolved_path).exists():
        candidate = Path(resolved_path)
        if not candidate.is_absolute():
            matches = list(Path(".").rglob(candidate.name))
            if matches:
                resolved_path = str(matches[0].resolve())

    result = await asyncio.to_thread(
        run_two_mode_agent,
        instruction=req.instruction,
        file_path=resolved_path,
        test_cases=req.test_cases if req.test_cases else None,
    )
    _api_log.info("[API] /agent/two-mode — done, mode=%s", result.get("mode"))
    return {"status": "ok", **result}


# ── Curriculum generation endpoints (full chapter at once) ───────────────

_curriculum_gen_state: dict = {"status": "idle", "step": 0, "total": 0, "message": ""}

class GenerateChapterRequest(BaseModel):
    subject_id: str
    chapter_id: str
    is_math: bool = True
    language: str = "python"

@app.post("/curriculum/generate")
async def curriculum_generate(req: GenerateChapterRequest):
    """Generate an entire chapter of problems (persist to JSON). Long-running — poll status."""
    from tutor.tutor import get_generated_chapter
    global _curriculum_gen_state

    # Return cached if already exists
    existing = get_generated_chapter(req.subject_id, req.chapter_id)
    if existing:
        return {"status": "ok", "cached": True, **existing}

    if _curriculum_gen_state["status"] == "running":
        return {"status": "already_running"}

    _curriculum_gen_state = {"status": "running", "step": 0, "total": 0, "message": "Starting..."}

    def _progress(step, total, message):
        _curriculum_gen_state["step"] = step
        _curriculum_gen_state["total"] = total
        _curriculum_gen_state["message"] = message

    async def _run():
        global _curriculum_gen_state
        try:
            from tutor.tutor import generate_full_chapter
            result = await asyncio.to_thread(
                generate_full_chapter,
                subject_id=req.subject_id,
                chapter_id=req.chapter_id,
                is_math=req.is_math,
                language=req.language,
                progress_callback=_progress,
            )
            _curriculum_gen_state = {"status": "done", "step": 0, "total": 0, "message": "Complete", "result": result}
        except Exception as e:
            _curriculum_gen_state = {"status": "error", "step": 0, "total": 0, "message": str(e)}

    asyncio.create_task(_run())
    return {"status": "started"}

@app.get("/curriculum/generate/status")
async def curriculum_generate_status():
    """Poll for progress of chapter generation."""
    return _curriculum_gen_state

@app.get("/curriculum/chapter/{subject_id}/{chapter_id}")
async def curriculum_get_chapter(subject_id: str, chapter_id: str):
    """Get a generated chapter's problems."""
    from tutor.tutor import get_generated_chapter
    data = get_generated_chapter(subject_id, chapter_id)
    if not data:
        return {"status": "not_generated"}
    return {"status": "ok", **data}

@app.get("/curriculum/chapter/{subject_id}/{chapter_id}/{problem_index}")
async def curriculum_get_problem(subject_id: str, chapter_id: str, problem_index: int):
    """Get a specific problem from a generated chapter."""
    from tutor.tutor import get_chapter_problem
    result = get_chapter_problem(subject_id, chapter_id, problem_index)
    if "error" in result:
        return {"status": "error", **result}
    return {"status": "ok", "problem": result}


# ── TTS Kokoro endpoint ────────────────────────────────────────────────

class TTSKokoroRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0


@app.post("/voice/tts")
async def voice_tts_kokoro(req: TTSKokoroRequest):
    """Text-to-speech using Kokoro ONNX. Returns WAV audio file."""
    _api_log.info("[API] /voice/tts — text=%d chars, voice=%s", len(req.text), req.voice)

    def _generate():
        try:
            from kokoro_onnx import Kokoro
            import soundfile as sf
            import numpy as np
            import tempfile
        except ImportError:
            return None, "kokoro-onnx not installed. Run: pip install kokoro-onnx"

        kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        samples, sample_rate = kokoro.create(
            req.text,
            voice=req.voice,
            speed=req.speed,
            lang="en-us"
        )

        if samples is None or len(samples) == 0:
            return None, "No audio generated"

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, samples, sample_rate)
        return tmp.name, None

    try:
        wav_path, error = await asyncio.to_thread(_generate)
        if error:
            return {"status": "error", "error": error}

        from starlette.responses import FileResponse
        return FileResponse(wav_path, media_type="audio/wav", filename="tts_output.wav")
    except Exception as e:
        return {"status": "error", "error": str(e)}



# ── Image generation endpoint ────────────────────────────────────────────

from diffusers import StableDiffusionXLPipeline
import torch

MODEL_PATH = "C:/Users/Pog Pete/Desktop/projects/Aion/models/illustriousXL_v01.safetensors"
assert Path(MODEL_PATH).exists(), f"Model not found at: {MODEL_PATH}"

_sd_pipe = None

def _get_sd_pipe():
    global _sd_pipe
    if _sd_pipe is None:
        _sd_pipe = StableDiffusionXLPipeline.from_single_file(
            MODEL_PATH,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")
        _sd_pipe.enable_attention_slicing()
    return _sd_pipe


class ImageGenRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024


@app.post("/generate/image")
async def generate_image(req: ImageGenRequest):
    _api_log.info("[API] /generate/image — prompt=%s", req.prompt[:60])

    # def _expand_prompt(raw: str) -> str:
    #     # Ask Ollama to expand the prompt
    #     import httpx
    #     resp = httpx.post("http://localhost:11434/api/generate", json={
    #         "model": "qwen3",
    #         "prompt": f"Expand this image generation prompt into a detailed, descriptive prompt for Stable Diffusion XL. Return only the prompt, no explanation: '{raw}'",
    #         "stream": False,
    #     }, timeout=30.0)
    #     if resp.status_code == 200:
    #         return resp.json().get("response", raw).strip()
    #     return raw

    def _generate():
        try:
            output_dir_path = Path(OUTPUT_DIR)
            output_dir_path.mkdir(parents=True, exist_ok=True)

            pipe = _get_sd_pipe()

            # Pony requires these prefix tags
            full_prompt = f"rating_explicit, amazing quality, sexy,{req.prompt}"
            full_negative = "score_4, score_3, score_2, score_1, " + \
                            "blurry, low quality, deformed, bad anatomy, watermark, bad hands, text, error, cropped, worst quality,bad quality,worst quality,worst detail,sketch,censored, artist os.name,signature, watermark,patreon username, patreon logo,pov, doorway, door, tattoo"

            image = pipe(
                full_prompt,
                width=req.width,
                height=req.height,
                num_inference_steps=35,
                guidance_scale=8.0,
                negative_prompt=full_negative,
            ).images[0]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = str(output_dir_path / f"{timestamp}_{req.prompt[:20].replace(' ', '_')}.png")
            image.save(filename)
            _api_log.info("[IMAGE GEN] Saved to: %s", filename)
            return filename, None
        except Exception as e:
            _api_log.error("[IMAGE GEN] Error: %s", str(e))
            return None, str(e)


    img_path, error = await asyncio.to_thread(_generate)
    if error:
        return {"status": "error", "error": error}

    from starlette.responses import FileResponse
    return FileResponse(img_path, media_type="image/png", filename="generated.png")



# ── Math visualization compute endpoint ──────────────────────────────────

@app.get("/visualizations/modules")
async def visualization_modules_list():
    """Get all visualization modules grouped by category (for frontend tabs)."""
    from tutor.math_visualizations import get_visualization_modules
    return {"status": "ok", "categories": get_visualization_modules()}


class QuadraticRequest(BaseModel):
    a: float = 1
    b: float = 0
    c: float = -4

@app.post("/math/quadratic")
async def math_quadratic(req: QuadraticRequest):
    """Compute quadratic formula: roots, vertex, parabola points, steps."""
    from tutor.math_visualizations import compute_quadratic
    result = compute_quadratic(a=req.a, b=req.b, c=req.c)
    return {"status": "ok", **result}


class ScientificCalcRequest(BaseModel):
    expression: str

@app.post("/math/scientific")
async def math_scientific(req: ScientificCalcRequest):
    """Evaluate a scientific expression (trig, integrals, derivatives, etc.)."""
    _api_log.info("[API] /math/scientific — expr=%s", req.expression[:60])
    from tutor.math_visualizations import compute_scientific
    result = await asyncio.to_thread(compute_scientific, req.expression)
    return {"status": "ok", **result}
