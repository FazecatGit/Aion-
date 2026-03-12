"""
Multi-Agent Orchestration via LangGraph.

Workflow:
  1. Planner    — decomposes a complex task into sub-steps
  2. Strategist — deep algorithmic analysis: identifies the correct approach,
                  produces pseudocode, and queries RAG for reference patterns
  3. CodeAgent  — executes each step (edit_code) guided by the strategy
  4. Critic     — verifies output, flags issues
  5. Router     — if critic fails, feeds issues back to CodeAgent for retry

Graph:
  plan → strategist → execute → critique ──(PASS)──→ done
                                    │
                                    └──(FAIL)──→ discuss → execute (retry)
"""

import difflib
import logging
import os
import re
from typing import TypedDict, Annotated, List, Dict, Optional

from langgraph.graph import StateGraph, END

from langchain_ollama import OllamaLLM

from brain.config import LLM_MODEL, LANG_FENCE, LANG_NAMES
from agent.orchestration import plan_task, format_plan_for_prompt, critique_code, build_critic_feedback_note

logger = logging.getLogger("multi_agent")


# ═══════════════════════════════════════════════════════════════════════════════
# STATE SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

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
    # Strategist output
    strategy: str  # detailed algorithmic strategy + pseudocode from strategist agent
    # Agent discussion
    discussion_log: List[str]  # conversation between agents when stuck
    # Test-driven execution
    test_cases: Optional[List[Dict]]    # [{input, expected}] — enables execution-based critique
    test_results: Optional[List[Dict]]  # [{input, expected, actual, passed, error}]


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH NODES
# ═══════════════════════════════════════════════════════════════════════════════

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
        steps = plan.get("steps", [])
        logger.info("[PLAN] %d steps, approach: %s", len(steps), plan.get("approach", "")[:200])
        for i, step in enumerate(steps, 1):
            logger.info("[PLAN]   Step %d: %s", i, step[:300])
        return {"plan": plan, "plan_text": plan_text}
    except Exception as e:
        logger.warning("[PLAN] Failed: %s", e)
        return {"plan": {"steps": [state["instruction"]], "approach": "", "edge_cases": []}, "plan_text": ""}


def strategist_node(state: AgentState) -> dict:
    """Strategist agent: deep algorithmic analysis before code is written.

    The strategist sits between the planner and the coder. It:
      1. Analyses what algorithm class / data structure the problem requires
      2. Queries RAG for relevant algorithm patterns and reference material
      3. Produces concrete pseudocode with state definitions and transitions
      4. Traces through the first test case to verify correctness

    The output is a detailed strategy string that the coder can follow
    mechanically — the coder doesn't need to independently derive the algorithm.
    """
    instruction = state["instruction"]
    source = state["original_source"]
    ext = state["ext"]
    plan = state.get("plan", {})
    approach = plan.get("approach", "")
    edge_cases = plan.get("edge_cases", [])
    test_cases = state.get("test_cases")

    lang = LANG_FENCE.get(ext, ext.lstrip('.'))
    lang_name = LANG_NAMES.get(ext, ext.lstrip('.').upper() if ext else 'code')

    llm = OllamaLLM(model=LLM_MODEL, temperature=0.1)

    logger.info("[STRATEGIST] Analyzing problem for algorithmic approach...")

    # ── Query RAG for relevant algorithm reference material ─────────
    rag_section = ""
    try:
        from brain.fast_search import fast_topic_search
        # Build a query targeting algorithm patterns, not just the language
        rag_query = f"algorithm {approach} dynamic programming combinatorics {instruction[:150]}"
        from brain.config import EXT_TO_TOPIC
        _topics = ["algorithms", "clean-code"]
        _lang_topic = EXT_TO_TOPIC.get(ext)
        if _lang_topic:
            _topics.append(_lang_topic)
        rag_results = fast_topic_search(
            rag_query,
            top_k=5,
            rerank_method="cross_encoder",
            topic_filter=_topics,
        )
        if rag_results:
            rag_chunks = []
            for doc in rag_results[:3]:
                chunk = doc.page_content[:600]
                if chunk.strip():
                    rag_chunks.append(chunk)
            if rag_chunks:
                rag_section = (
                    "\n\nREFERENCE MATERIAL (from algorithm textbooks):\n"
                    + "\n---\n".join(rag_chunks)
                    + "\n"
                )
                logger.info("[STRATEGIST] RAG: injected %d reference chunks", len(rag_chunks))
    except Exception as e:
        logger.debug("[STRATEGIST] RAG query failed: %s", e)

    # ── Build test case context ─────────────────────────────────────
    test_section = ""
    if test_cases:
        test_lines = [
            f"  Test {i+1}: input=({tc['input']}) → expected={tc['expected']}"
            for i, tc in enumerate(test_cases[:5])
        ]
        test_section = "\n\nTEST CASES:\n" + "\n".join(test_lines)

    # ── Build edge case context from planner ────────────────────────
    edge_section = ""
    if edge_cases:
        edge_section = "\n\nEDGE CASES TO HANDLE:\n" + "\n".join(f"  - {ec}" for ec in edge_cases)

    # ── Main strategist prompt ──────────────────────────────────────
    prompt = (
        f"You are the STRATEGIST agent — an expert algorithm designer. "
        f"Your job is to figure out the CORRECT algorithm before any code is written.\n\n"
        f"PROBLEM:\n{instruction}\n\n"
        f"LANGUAGE: {lang_name} ({ext})\n\n"
        f"CURRENT CODE (stub or broken attempt):\n```{lang}\n{source[:4000]}\n```\n"
        f"{test_section}"
        f"{edge_section}"
        f"{rag_section}\n\n"
        f"PLANNER'S APPROACH: {approach}\n\n"
        f"Analyze this problem step by step:\n\n"
        f"1. PROBLEM CLASSIFICATION: What type of problem is this? "
        f"(DP, greedy, graph, combinatorics, math, etc.)\n\n"
        f"2. KEY INSIGHT: What is the non-obvious observation that makes this solvable? "
        f"What constraint must the algorithm enforce?\n\n"
        f"3. STATE DEFINITION: If DP, define the state precisely. "
        f"What does dp[i][j] represent? What are the dimensions?\n\n"
        f"4. TRANSITIONS: How do you move between states? "
        f"Write the recurrence relation. If there's a prefix-sum or sliding-window trick, explain it.\n\n"
        f"5. BASE CASES: What are the initial values?\n\n"
        f"6. PSEUDOCODE: Write clear pseudocode (NOT {lang_name} code — just pseudocode) "
        f"that a coder can translate directly.\n\n"
        f"7. TRACE: Walk through the first test case with your algorithm to verify it produces "
        f"the expected output.\n\n"
        f"Be CONCRETE — specify array dimensions, loop bounds, exact formulas. "
        f"Do NOT be vague. The coder will follow this mechanically."
    )

    try:
        strategy = llm.invoke(prompt).strip()
        # Truncate if excessively long (shouldn't be, but safety)
        if len(strategy) > 4000:
            strategy = strategy[:4000] + "\n... (truncated)"
        logger.info("[STRATEGIST] Strategy produced (%d chars)", len(strategy))
        logger.info("[STRATEGIST] Preview: %s", strategy[:200])
        return {"strategy": strategy}
    except Exception as e:
        logger.warning("[STRATEGIST] Failed: %s", e)
        return {"strategy": ""}


def execute_node(state: AgentState) -> dict:
    """Code agent: apply edits based on plan + strategy + any critic feedback.

    When test_cases are available, uses the iterative fix_with_tests loop
    (debug trace + step verification + strategy pivot) instead of one-shot edit.
    """
    from agent.code_agent import CodeAgent

    instruction = state["instruction"]
    critic_feedback = state.get("critic_feedback", "")
    strategy = state.get("strategy", "")

    # Augment instruction with plan, strategy, and critic feedback
    augmented = instruction
    if state.get("plan_text"):
        augmented = f"{instruction}\n\nPLAN:\n{state['plan_text']}"
    if strategy:
        augmented = f"{augmented}\n\nSTRATEGY (follow this algorithm closely):\n{strategy}"
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

        # ── Test-driven path: use iterative fix_with_tests ──────────────
        if state.get("test_cases"):
            logger.info("[EXECUTE] Test cases available — using fix_with_tests loop")
            # Write current source to disk for fix_with_tests
            try:
                with open(state["file_path"], "w", encoding="utf-8") as f:
                    f.write(state["current_source"])
            except Exception:
                pass

            result = agent.fix_with_tests(
                path=state["file_path"],
                instruction=augmented,
                test_cases=state["test_cases"],
                max_retries=2,  # fewer retries per-attempt since outer loop also retries
                task_mode=state.get("task_mode", "solve"),
                session_id=state.get("session_id"),
            )
            test_results = result.get("test_results", [])
            return {
                "current_source": result.get("final_source", state["current_source"]),
                "diff": result.get("diff", ""),
                "explanation": result.get("explanation", ""),
                "citations": result.get("citations", []),
                "test_results": test_results,
                "attempt": state["attempt"] + 1,
            }

        # ── Standard path: one-shot edit_code ──────────────────────────
        result = agent.edit_code(
            path=state["file_path"],
            instruction=augmented,
            dry_run=True,
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
    """Critic agent: verify the code against requirements.

    When test_results are available (from fix_with_tests), uses execution-based
    verdict instead of LLM-only review \u2014 real test data beats guessing.
    """
    if state.get("error"):
        return {"verdict": "FAIL", "critic_feedback": f"Execution error: {state['error']}"}

    if state["current_source"] == state["original_source"]:
        return {"verdict": "FAIL", "critic_feedback": "No changes were made to the code."}

    # \u2500\u2500 Execution-based critique (when test results exist) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    test_results = state.get("test_results")
    if test_results:
        pass_count = sum(1 for r in test_results if r.get("passed"))
        total = len(test_results)
        all_passed = pass_count == total

        if all_passed:
            logger.info("[CRITIQUE] All %d tests pass \u2014 PASS", total)
            return {"verdict": "PASS", "critic_feedback": "", "test_results": test_results}

        # Build concrete feedback from actual failures
        failures = [r for r in test_results if not r.get("passed")]
        feedback_lines = [
            f"EXECUTION-BASED CRITIQUE ({pass_count}/{total} tests passing):\n"
        ]
        for i, r in enumerate(failures, 1):
            got = r.get("actual") or f"ERROR: {r.get('error', 'unknown')}"
            feedback_lines.append(
                f"  FAIL Test {i}: input={r['input']}  expected={r['expected']}  got={got}"
            )
        feedback_lines.append(
            "\nThese are REAL execution results, not guesses. "
            "Fix the specific logic errors causing wrong output."
        )
        feedback = "\n".join(feedback_lines)
        logger.info("[CRITIQUE] %d/%d tests pass \u2014 FAIL", pass_count, total)
        return {"verdict": "FAIL", "critic_feedback": feedback, "test_results": test_results}

    # LLM-based critique (no tests available) 
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
    logger.info("[ROUTER] Critic failed, routing to agent discussion (attempt %d/%d)", state["attempt"], state["max_attempts"])
    return "discuss"


def discuss_node(state: AgentState) -> dict:
    """Agent discussion: Planner and Critic debate the problem when stuck.

    The Planner proposes a new approach, the Critic evaluates it, and they
    iterate to reach consensus before the next execution attempt.
    """
    logger.info("[DISCUSS] Agents debating the problem (attempt %d)...", state["attempt"])

    llm = OllamaLLM(model=LLM_MODEL, temperature=0.3)
    lang = LANG_FENCE.get(state["ext"], state["ext"].lstrip('.'))
    discussion = list(state.get("discussion_log") or [])

    critic_feedback = state.get("critic_feedback", "No specific feedback")
    current_code = state.get("current_source", "")[:4000]
    instruction = state["instruction"]
    strategy = state.get("strategy", "")

    # Build test results section if available (gives agents real execution data)
    test_section = ""
    test_results = state.get("test_results")
    if test_results:
        test_lines = []
        for r in test_results:
            status = "PASS" if r.get("passed") else "FAIL"
            got = r.get("actual") or f"ERROR: {r.get('error', 'N/A')}"
            test_lines.append(f"  {status}: input={r['input']} expected={r['expected']} got={got}")
        test_section = "\n\nACTUAL TEST EXECUTION RESULTS:\n" + "\n".join(test_lines)

    # Include strategist's analysis so discussion agents can reference it
    strategy_section = ""
    if strategy:
        strategy_section = f"\n\nORIGINAL STRATEGY (from strategist agent):\n{strategy[:1500]}\n"

    # Round 1: Planner proposes a new approach based on critic feedback
    planner_prompt = (
        f"You are the PLANNER agent in a multi-agent coding system.\n\n"
        f"TASK: {instruction}\n"
        f"LANGUAGE: {lang}\n\n"
        f"The CRITIC agent rejected the previous attempt with this feedback:\n"
        f"{critic_feedback}\n"
        f"{test_section}\n"
        f"{strategy_section}\n\n"
        f"Current code:\n```{lang}\n{current_code}\n```\n\n"
        f"The original strategy may have been wrong or the coder may have deviated from it.\n"
        f"Propose a CORRECTED approach. Be specific about:\n"
        f"1. What went wrong in the previous attempt\n"
        f"2. What the correct algorithm/state definition should be\n"
        f"3. Specific recurrence relations or transitions needed\n\n"
        f"Keep your response concise (3-5 sentences)."
    )
    planner_response = llm.invoke(planner_prompt).strip()
    discussion.append(f"[PLANNER] {planner_response}")
    logger.info("[DISCUSS] Planner: %s", planner_response[:300])

    # Round 2: Critic evaluates the planner's proposal
    critic_prompt = (
        f"You are the CRITIC agent in a multi-agent coding system.\n\n"
        f"TASK: {instruction}\n"
        f"LANGUAGE: {lang}\n"
        f"{test_section}\n"
        f"{strategy_section}\n\n"
        f"The PLANNER proposed this approach:\n{planner_response}\n\n"
        f"Previous issues: {critic_feedback}\n\n"
        f"Evaluate this proposal:\n"
        f"1. Will it address the issues found?\n"
        f"2. Does it align with the original strategy or improve on it?\n"
        f"3. Are there any risks or edge cases the planner missed?\n\n"
        f"If you agree with the approach, say 'AGREED' and elaborate briefly.\n"
        f"If not, explain what needs to change. Keep concise (3-5 sentences)."
    )
    critic_response = llm.invoke(critic_prompt).strip()
    discussion.append(f"[CRITIC] {critic_response}")
    logger.info("[DISCUSS] Critic: %s", critic_response[:300])

    # Round 3: If critic disagreed, planner adjusts
    if "AGREED" not in critic_response.upper():
        adjust_prompt = (
            f"You are the PLANNER agent. The CRITIC raised concerns about your proposal.\n\n"
            f"Your proposal: {planner_response}\n"
            f"Critic's feedback: {critic_response}\n\n"
            f"Adjust your approach to address the critic's concerns. "
            f"Provide a final, concrete plan. Keep concise (3-5 sentences)."
        )
        final_plan = llm.invoke(adjust_prompt).strip()
        discussion.append(f"[PLANNER revised] {final_plan}")
        logger.info("[DISCUSS] Planner revised: %s", final_plan[:300])
        consensus = final_plan
    else:
        consensus = planner_response

    # Synthesize the discussion into updated critic_feedback for the executor
    enhanced_feedback = (
        f"{critic_feedback}\n\n"
        f"AGENT DISCUSSION CONSENSUS:\n{consensus}"
    )

    # ── Strategist escalation on repeated failures ──────────────────
    # If we've failed 2+ times, the algorithm itself may be wrong.
    # Re-invoke the strategist with the full failure context so it can
    # propose a fundamentally different approach rather than patching.
    if state["attempt"] >= 2:
        logger.info("[DISCUSS] Escalating to strategist (attempt %d)", state["attempt"])
        try:
            llm_strat = OllamaLLM(model=LLM_MODEL, temperature=0.2)
            lang_name = LANG_NAMES.get(state["ext"], state["ext"])
            lang = LANG_FENCE.get(state["ext"], state["ext"].lstrip('.'))

            failed_summary = "\n".join(discussion[-3:])  # last 3 messages
            test_section = ""
            if test_results:
                test_lines = [
                    f"  {('PASS' if r.get('passed') else 'FAIL')}: "
                    f"input={r['input']} expected={r['expected']} got={r.get('actual', r.get('error', 'N/A'))}"
                    for r in test_results
                ]
                test_section = "\nACTUAL TEST RESULTS:\n" + "\n".join(test_lines)

            escalation_prompt = (
                f"You are the STRATEGIST agent being re-consulted because the coder has FAILED "
                f"{state['attempt']} times on this problem.\n\n"
                f"PROBLEM: {instruction}\n"
                f"LANGUAGE: {lang_name}\n\n"
                f"CURRENT (BROKEN) CODE:\n```{lang}\n{state.get('current_source', '')[:4000]}\n```\n"
                f"{test_section}\n\n"
                f"PREVIOUS STRATEGY (which led to failure):\n{strategy[:1000]}\n\n"
                f"AGENT DISCUSSION:\n{failed_summary}\n\n"
                f"The previous approach is WRONG. You must propose a FUNDAMENTALLY DIFFERENT "
                f"algorithm — not a patch. Common escape routes:\n"
                f"- If using combinatorics → switch to DP\n"
                f"- If using brute force → switch to greedy or binary search\n"
                f"- If using BFS → switch to DFS with memoization\n"
                f"- If using a formula → switch to simulation/DP\n\n"
                f"1. WHAT WAS WRONG: Why the previous approach cannot work\n"
                f"2. CORRECT ALGORITHM CLASS: (DP / greedy / graph / etc.)\n"
                f"3. STATE DEFINITION: Precise dp[i][j] or equivalent\n"
                f"4. RECURRENCE: Exact formula\n"
                f"5. PSEUDOCODE: Clear enough to translate directly\n"
                f"6. TRACE: Walk through first test case\n"
            )
            new_strategy = llm_strat.invoke(escalation_prompt).strip()
            if len(new_strategy) > 4000:
                new_strategy = new_strategy[:4000] + "\n... (truncated)"
            logger.info("[DISCUSS] Strategist escalation produced %d chars", len(new_strategy))

            enhanced_feedback += f"\n\nREVISED STRATEGY (from strategist re-consultation):\n{new_strategy}"
            return {
                "discussion_log": discussion,
                "critic_feedback": enhanced_feedback,
                "strategy": new_strategy,
            }
        except Exception as e:
            logger.warning("[DISCUSS] Strategist escalation failed: %s", e)

    return {
        "discussion_log": discussion,
        "critic_feedback": enhanced_feedback,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_agent_graph() -> StateGraph:
    """Build and compile the multi-agent LangGraph.

    Graph: plan → execute → critique ──(PASS)──→ done
                                │
                                └──(FAIL)──→ discuss → execute (retry)
    """
    graph = StateGraph(AgentState)

    graph.add_node("plan", plan_node)
    graph.add_node("strategist", strategist_node)
    graph.add_node("execute", execute_node)
    graph.add_node("critique", critique_node)
    graph.add_node("discuss", discuss_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "strategist")
    graph.add_edge("strategist", "execute")
    graph.add_edge("execute", "critique")
    graph.add_edge("discuss", "execute")  # after discussion, retry execution

    graph.add_conditional_edges(
        "critique",
        should_retry,
        {"discuss": "discuss", "done": END},
    )

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

# Cached compiled graph — reset on import to pick up graph changes
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


def _edit_context_files(
    instruction: str,
    main_file_path: str,
    main_new_source: str,
    context_file_paths: List[str],
    plan: Dict,
    task_mode: str,
    session_id: str,
) -> List[Dict]:
    """Run CodeAgent on each context file to produce edits informed by the main change."""
    from agent.code_agent import CodeAgent

    edits = []
    agent = CodeAgent(repo_path=".")

    # Build context about what changed in the main file
    original_main = _read_file_safe(main_file_path) or ""
    main_diff_lines = list(difflib.unified_diff(
        original_main.splitlines(keepends=True),
        main_new_source.splitlines(keepends=True),
        fromfile=main_file_path,
        tofile=main_file_path,
    ))
    main_diff_text = "".join(main_diff_lines)[:3000]

    plan_steps = "\n".join(f"- {s}" for s in plan.get("steps", [])) if plan else ""

    # Determine if the user wants the same task applied to ALL context files
    # (cross-language implementation) vs just keeping them consistent.
    # Use three signals:
    #   1. Different file extensions (Go+CPP, Python+Java, etc.) — strong signal
    #   2. The plan mentions multiple files or languages explicitly
    #   3. LLM classification of the instruction intent (understands natural language)
    _has_cross_lang = False
    _context_exts = set()
    main_ext = os.path.splitext(main_file_path)[1].lower()
    for _cp in context_file_paths:
        _ce = os.path.splitext(_cp)[1].lower()
        if _ce != main_ext:
            _has_cross_lang = True
        _context_exts.add(_ce)

    _apply_to_all = _has_cross_lang  # different extensions → almost always cross-impl
    if not _apply_to_all and len(context_file_paths) > 0:
        # Use a fast LLM probe to decide if the user intends changes in all files
        try:
            _clf_llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)
            _file_list = ", ".join(os.path.basename(p) for p in context_file_paths)
            _clf_prompt = (
                f"A user gave this instruction to edit code:\n\"{instruction}\"\n\n"
                f"The main file is {os.path.basename(main_file_path)}.\n"
                f"Other files in the project: {_file_list}\n\n"
                "Does the user intend the same change to be applied to ALL these files, "
                "or only to the main file (with others kept consistent)?\n"
                "Answer ONLY with: ALL or MAIN"
            )
            _clf_result = _clf_llm.invoke(_clf_prompt).strip().upper()
            _apply_to_all = "ALL" in _clf_result
            logger.info("[CONTEXT_EDIT] LLM intent classification: %s → apply_to_all=%s",
                        _clf_result[:20], _apply_to_all)
        except Exception as e:
            logger.debug("[CONTEXT_EDIT] LLM classification failed: %s", e)

    for cf_path in context_file_paths:
        abs_cf = os.path.abspath(cf_path)
        if abs_cf == os.path.abspath(main_file_path):
            continue
        if not os.path.isfile(abs_cf):
            continue

        original_cf = _read_file_safe(abs_cf)
        if original_cf is None:
            continue

        cf_ext = os.path.splitext(abs_cf)[1].lower()
        main_ext = os.path.splitext(main_file_path)[1].lower()
        cf_lang = LANG_NAMES.get(cf_ext, cf_ext)
        main_lang = LANG_NAMES.get(main_ext, main_ext)

        # Build the augmented instruction based on whether this is a
        # cross-language implementation (same task, different language) or
        # a dependency-style edit (keep consistent with main file changes)
        if _apply_to_all or cf_ext != main_ext:
            # Cross-language or explicit multi-file: apply the SAME original
            # instruction to this file — the user wants the same task solved here
            augmented = (
                f"{instruction}\n\n"
                f"You are editing the {cf_lang} file ({os.path.basename(abs_cf)}).\n"
                f"The same task was already implemented in the {main_lang} file.\n"
                f"Here is what was done in the main file for reference:\n"
                f"```diff\n{main_diff_text}\n```\n\n"
            )
            if plan_steps:
                augmented += f"PLAN:\n{plan_steps}\n\n"
            augmented += (
                f"Implement the same solution in {cf_lang} in this file. "
                f"Apply edits ONLY to the relevant function — preserve all other code in the file."
            )
        else:
            # Same-language dependency: keep consistent with main file changes
            augmented = (
                f"{instruction}\n\n"
                f"MAIN FILE CHANGES ({main_file_path}):\n"
                f"```diff\n{main_diff_text}\n```\n\n"
            )
            if plan_steps:
                augmented += f"PLAN:\n{plan_steps}\n\n"
            augmented += (
                "Apply any necessary corresponding changes to this file "
                "to keep it consistent with the main file changes above. "
                "If no changes are needed, return the file unchanged."
            )

        # Determine task mode: if we're implementing in the main file,
        # also implement in context files
        cf_task_mode = task_mode
        if _apply_to_all and task_mode in ("solve", "auto"):
            cf_task_mode = "solve"

        try:
            result = agent.edit_code(
                path=abs_cf,
                instruction=augmented,
                dry_run=True,
                use_rag=_apply_to_all,  # use RAG for cross-file implementations
                task_mode=cf_task_mode,
                session_id=session_id,
            )

            if isinstance(result, dict):
                new_src = result.get("new_source", original_cf)
            else:
                new_src = result if isinstance(result, str) else original_cf

            # Restore original on disk (edit_code may have written)
            try:
                with open(abs_cf, "w", encoding="utf-8") as f:
                    f.write(original_cf)
            except Exception:
                pass

            if new_src != original_cf:
                diff_lines = list(difflib.unified_diff(
                    original_cf.splitlines(keepends=True),
                    new_src.splitlines(keepends=True),
                    fromfile=abs_cf,
                    tofile=abs_cf,
                ))
                edits.append({
                    "path": abs_cf,
                    "diff": "".join(diff_lines),
                    "new_source": new_src,
                    "explanation": result.get("explanation", "") if isinstance(result, dict) else "",
                })
                logger.info("[CONTEXT_EDIT] Produced edit for %s", abs_cf)
            else:
                logger.info("[CONTEXT_EDIT] No changes needed for %s", abs_cf)

        except Exception as e:
            logger.warning("[CONTEXT_EDIT] Failed for %s: %s", abs_cf, e)

    return edits


def run_multi_agent(
    instruction: str,
    file_path: str,
    task_mode: str = "auto",
    session_id: str = None,
    max_attempts: int = 3,
    include_related: bool = True,
    extra_context_files: List[str] = None,
    test_cases: Optional[List[Dict]] = None,
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
        "strategy": "",
        "critic_feedback": "",
        "attempt": 0,
        "max_attempts": max_attempts,
        "diff": "",
        "explanation": "",
        "citations": [],
        "verdict": "",
        "error": None,
        "related_files": related_files,
        "discussion_log": [],
        "test_cases": test_cases,
        "test_results": None,
    }

    graph = get_agent_graph()
    final_state = graph.invoke(initial_state)

    # Restore the original file on disk — the pipeline may have written
    # intermediate sources during retries. The user must approve via the
    # frontend before any changes are persisted.
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(source)
        logger.info("[MULTI_AGENT] Original file restored after pipeline")
    except Exception as e:
        logger.warning("[MULTI_AGENT] Could not restore original file: %s", e)

    new_source = final_state.get("current_source", source)
    has_changes = new_source != source

    # ── Edit context files if the instruction targets them ──────────────
    # When the user provides context_files, check if the planner/instruction
    # implies changes to those files too. Run the code agent on each one.
    context_file_edits = []
    if extra_context_files and has_changes:
        context_file_edits = _edit_context_files(
            instruction=instruction,
            main_file_path=file_path,
            main_new_source=new_source,
            context_file_paths=extra_context_files,
            plan=final_state.get("plan", {}),
            task_mode=task_mode,
            session_id=session_id,
        )

    return {
        "status": "pending_review" if has_changes else "ok",
        "diff": final_state.get("diff", ""),
        "explanation": final_state.get("explanation", ""),
        "citations": final_state.get("citations", []),
        "verdict": final_state.get("verdict", ""),
        "plan": final_state.get("plan", {}),
        "attempts": final_state.get("attempt", 0),
        "error": final_state.get("error"),
        "related_files": [rf["path"] for rf in related_files],
        "new_source": new_source,
        "file_path": file_path,
        "strategy": final_state.get("strategy", ""),
        "critic_feedback": final_state.get("critic_feedback", ""),
        "discussion_log": final_state.get("discussion_log", []),
        "context_file_edits": context_file_edits,
        "test_results": final_state.get("test_results"),
    }
