"""
Multi-Agent Orchestration via LangGraph.

Workflow:
  1. Planner  — decomposes a complex task into sub-steps
  2. CodeAgent — executes each step (edit_code)
  3. Critic   — verifies output, flags issues
  4. Router   — if critic fails, feeds issues back to CodeAgent for retry

Graph:
  plan → execute → critique ──(PASS)──→ done
                       │
                       └──(FAIL)──→ execute (retry with critic feedback, max 2)
"""

import logging
import os
from typing import TypedDict, Annotated, List, Dict, Optional

from langgraph.graph import StateGraph, END

from langchain_ollama import OllamaLLM

from brain.config import LLM_MODEL, LANG_FENCE
from agent.orchestration import plan_task, format_plan_for_prompt, critique_code, build_critic_feedback_note

logger = logging.getLogger("multi_agent")


# ── State schema ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Inputs
    instruction: str
    file_path: str
    task_mode: str
    session_id: Optional[str]
    # Working state
    original_source: str
    current_source: str
    ext: str
    plan: Dict
    plan_text: str
    critic_feedback: str
    attempt: int
    max_attempts: int
    # Outputs
    diff: str
    explanation: str
    citations: List[str]
    verdict: str
    error: Optional[str]
    # Multi-file support
    related_files: List[Dict]  # [{path, source}]


# ── Node functions ───────────────────────────────────────────────────────────

def plan_node(state: AgentState) -> dict:
    """Planner agent: decompose instruction into sub-steps."""
    logger.info("[PLAN] Decomposing task: %s", state["instruction"][:100])
    ext = state["ext"]
    source = state["original_source"]

    try:
        llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)
        plan = plan_task(
            instruction=state["instruction"],
            source=source,
            ext=ext,
            llm=llm,
        )
        plan_text = format_plan_for_prompt(plan)
        logger.info("[PLAN] %d steps, approach: %s", len(plan.get("steps", [])), plan.get("approach", "")[:80])
        return {"plan": plan, "plan_text": plan_text}
    except Exception as e:
        logger.warning("[PLAN] Failed: %s", e)
        return {"plan": {"steps": [state["instruction"]], "approach": "", "edge_cases": []}, "plan_text": ""}


def execute_node(state: AgentState) -> dict:
    """Code agent: apply edits based on plan + any critic feedback."""
    from agent.code_agent import CodeAgent

    instruction = state["instruction"]
    critic_feedback = state.get("critic_feedback", "")

    # Augment instruction with plan and critic feedback
    augmented = instruction
    if state.get("plan_text"):
        augmented = f"{instruction}\n\nPLAN:\n{state['plan_text']}"
    if critic_feedback:
        augmented = f"{augmented}\n\nPREVIOUS ATTEMPT ISSUES:\n{critic_feedback}\nFix the issues above."

    # Add related file context for multi-file awareness
    if state.get("related_files"):
        related_context = "\n\nRELATED FILES (read-only context):\n"
        for rf in state["related_files"]:
            related_context += f"\n--- {rf['path']} ---\n{rf['source'][:2000]}\n"
        augmented += related_context

    logger.info("[EXECUTE] Attempt %d — running code agent", state["attempt"])

    # On retry attempts, write the current_source to disk so edit_code reads the
    # updated file rather than the stale original.
    if state["attempt"] > 0 and state["current_source"] != state["original_source"]:
        try:
            with open(state["file_path"], "w", encoding="utf-8") as f:
                f.write(state["current_source"])
            logger.info("[EXECUTE] Wrote updated source to disk for retry")
        except Exception as we:
            logger.warning("[EXECUTE] Could not write updated source: %s", we)

    try:
        agent = CodeAgent(repo_path=".")
        result = agent.edit_code(
            path=state["file_path"],
            instruction=augmented,
            dry_run=False,  # Apply changes to disk so retries see the updated file
            use_rag=True,
            task_mode=state.get("task_mode", "auto"),
            session_id=state.get("session_id"),
        )

        if isinstance(result, dict):
            return {
                "current_source": result.get("new_source", state["current_source"]),
                "diff": result.get("diff", ""),
                "explanation": result.get("explanation", ""),
                "citations": result.get("citations", []),
                "attempt": state["attempt"] + 1,
            }
        else:
            return {
                "current_source": result if isinstance(result, str) else state["current_source"],
                "attempt": state["attempt"] + 1,
            }
    except Exception as e:
        logger.error("[EXECUTE] Failed: %s", e)
        return {"error": str(e), "attempt": state["attempt"] + 1}


def critique_node(state: AgentState) -> dict:
    """Critic agent: verify the code against requirements."""
    if state.get("error"):
        return {"verdict": "FAIL", "critic_feedback": f"Execution error: {state['error']}"}

    if state["current_source"] == state["original_source"]:
        return {"verdict": "FAIL", "critic_feedback": "No changes were made to the code."}

    logger.info("[CRITIQUE] Reviewing code changes...")

    try:
        llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)
        critique = critique_code(
            instruction=state["instruction"],
            original_source=state["original_source"],
            new_source=state["current_source"],
            ext=state["ext"],
            llm=llm,
        )

        verdict = critique["verdict"]
        logger.info("[CRITIQUE] Verdict: %s (confidence: %.1f)", verdict, critique["confidence"])

        if verdict == "PASS" or (verdict == "PARTIAL" and critique["confidence"] >= 0.7):
            return {"verdict": "PASS", "critic_feedback": ""}

        feedback = build_critic_feedback_note(critique)
        return {"verdict": "FAIL", "critic_feedback": feedback}

    except Exception as e:
        logger.warning("[CRITIQUE] Failed: %s", e)
        # If critic fails, assume pass (don't block)
        return {"verdict": "PASS", "critic_feedback": ""}


def should_retry(state: AgentState) -> str:
    """Router: decide whether to retry or finish."""
    if state.get("verdict") == "PASS":
        return "done"
    if state["attempt"] >= state["max_attempts"]:
        logger.warning("[ROUTER] Max attempts reached (%d), stopping", state["max_attempts"])
        return "done"
    logger.info("[ROUTER] Critic failed, routing back to execute (attempt %d/%d)", state["attempt"], state["max_attempts"])
    return "retry"


# ── Build the graph ──────────────────────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    """Build and compile the multi-agent LangGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("critique", critique_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "critique")

    graph.add_conditional_edges(
        "critique",
        should_retry,
        {"retry": "execute", "done": END},
    )

    return graph.compile()


# ── Public API ───────────────────────────────────────────────────────────────

# Cached compiled graph (safe to reuse)
_compiled_graph = None


def get_agent_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_agent_graph()
    return _compiled_graph


def _read_file_safe(path: str) -> Optional[str]:
    """Read file contents safely. Returns None on error, empty string for empty files."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return None


def _find_related_files(file_path: str, instruction: str) -> List[Dict]:
    """
    Find files that might be related to the current edit.
    Looks for imports, includes, and explicit file mentions in the instruction.
    """
    import re

    related = []
    source = _read_file_safe(file_path)
    base_dir = os.path.dirname(file_path) or "."
    ext = os.path.splitext(file_path)[1].lower()

    # Extract includes/imports from source
    import_paths = set()

    if ext in (".py",):
        # Python: from X import Y, import X
        for match in re.finditer(r'(?:from|import)\s+([\w.]+)', source):
            mod = match.group(1).replace('.', os.sep) + '.py'
            import_paths.add(mod)

    elif ext in (".go",):
        # Go: check files in same package directory
        for f in os.listdir(base_dir):
            if f.endswith(".go") and os.path.join(base_dir, f) != file_path:
                import_paths.add(os.path.join(base_dir, f))

    elif ext in (".cpp", ".c", ".h", ".hpp"):
        # C/C++: #include "local.h"
        for match in re.finditer(r'#include\s+"([^"]+)"', source):
            import_paths.add(os.path.join(base_dir, match.group(1)))

    elif ext in (".ts", ".tsx", ".js", ".jsx"):
        # JS/TS: import ... from './path'
        for match in re.finditer(r"from\s+['\"](\./[^'\"]+)['\"]", source):
            p = match.group(1)
            for try_ext in ["", ".ts", ".tsx", ".js", ".jsx"]:
                candidate = os.path.join(base_dir, p + try_ext)
                if os.path.isfile(candidate):
                    import_paths.add(candidate)
                    break

    # Also check for file paths explicitly mentioned in the instruction
    for match in re.finditer(r'[\w/\\]+\.\w{1,4}', instruction):
        candidate = match.group(0)
        if os.path.isfile(candidate):
            import_paths.add(candidate)
        elif os.path.isfile(os.path.join(base_dir, candidate)):
            import_paths.add(os.path.join(base_dir, candidate))

    # Read related file sources
    for p in import_paths:
        abs_path = os.path.abspath(p)
        if os.path.isfile(abs_path) and abs_path != os.path.abspath(file_path):
            content = _read_file_safe(abs_path)
            if content:
                related.append({"path": abs_path, "source": content})
                if len(related) >= 5:  # cap at 5 related files
                    break

    return related


def run_multi_agent(
    instruction: str,
    file_path: str,
    task_mode: str = "auto",
    session_id: str = None,
    max_attempts: int = 3,
    include_related: bool = True,
    extra_context_files: List[str] = None,
) -> Dict:
    """
    Run the full planner → executor → critic pipeline.

    Returns:
        {
            "status": "ok" | "error",
            "diff": str,
            "explanation": str,
            "citations": list,
            "verdict": str,
            "plan": dict,
            "attempts": int,
            "related_files": list[str],  # paths of related files read
        }
    """
    source = _read_file_safe(file_path)
    if source is None:
        return {"status": "error", "error": f"Cannot read file: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()

    # Find related files for multi-file context
    related_files = []
    if include_related:
        related_files = _find_related_files(file_path, instruction)

    # Add explicitly provided context files
    if extra_context_files:
        seen_paths = {os.path.abspath(rf["path"]) for rf in related_files}
        for cf_path in extra_context_files:
            abs_cf = os.path.abspath(cf_path)
            if abs_cf not in seen_paths and abs_cf != os.path.abspath(file_path) and os.path.isfile(abs_cf):
                content = _read_file_safe(abs_cf)
                if content is not None:
                    related_files.append({"path": abs_cf, "source": content})
                    seen_paths.add(abs_cf)
                    if len(related_files) >= 10:  # cap at 10 related files
                        break

    if related_files:
            logger.info("[MULTI_AGENT] Found %d related files: %s",
                        len(related_files), [rf["path"] for rf in related_files])

    initial_state: AgentState = {
        "instruction": instruction,
        "file_path": file_path,
        "task_mode": task_mode,
        "session_id": session_id,
        "original_source": source,
        "current_source": source,
        "ext": ext,
        "plan": {},
        "plan_text": "",
        "critic_feedback": "",
        "attempt": 0,
        "max_attempts": max_attempts,
        "diff": "",
        "explanation": "",
        "citations": [],
        "verdict": "",
        "error": None,
        "related_files": related_files,
    }

    graph = get_agent_graph()
    final_state = graph.invoke(initial_state)

    return {
        "status": "ok" if not final_state.get("error") else "error",
        "diff": final_state.get("diff", ""),
        "explanation": final_state.get("explanation", ""),
        "citations": final_state.get("citations", []),
        "verdict": final_state.get("verdict", ""),
        "plan": final_state.get("plan", {}),
        "attempts": final_state.get("attempt", 0),
        "error": final_state.get("error"),
        "related_files": [rf["path"] for rf in related_files],
        "new_source": final_state.get("current_source", source),
        "file_path": file_path,
    }
