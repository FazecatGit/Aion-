"""
Multi-agent orchestration: Planner + Critic for the code agent.

Planner  — decomposes a complex task into actionable sub-steps
Critic   — verifies agent output against requirements and RAG docs
"""

import logging
from typing import List, Dict, Optional

from langchain_ollama import OllamaLLM

from brain.config import LLM_MODEL, LANG_FENCE

logger = logging.getLogger("code_agent")


# ── Planner ──────────────────────────────────────────────────────────────────

def plan_task(
    instruction: str,
    source: str,
    ext: str,
    rag_context: str = "",
    llm: Optional[OllamaLLM] = None,
) -> Dict:
    """
    Decompose a complex instruction into ordered sub-steps.

    Returns:
        {
            "steps": ["step 1 description", "step 2 description", ...],
            "approach": "brief summary of chosen algorithm/strategy",
            "edge_cases": ["edge case 1", ...],
        }
    """
    if llm is None:
        llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)

    lang = LANG_FENCE.get(ext, ext.lstrip('.'))

    prompt = (
        f"You are a senior software engineer planning how to solve a coding task.\n\n"
        f"TASK: {instruction}\n\n"
        f"LANGUAGE: {lang}\n\n"
        f"CURRENT CODE:\n```{lang}\n{source[:3000]}\n```\n\n"
    )

    if rag_context:
        prompt += f"REFERENCE DOCUMENTATION:\n{rag_context[:2000]}\n\n"

    prompt += (
        "Create a step-by-step plan. Output EXACTLY this format:\n\n"
        "APPROACH: <one-line summary of the algorithm/strategy>\n\n"
        "STEPS:\n"
        "1. <specific actionable step>\n"
        "2. <specific actionable step>\n"
        "...\n\n"
        "EDGE_CASES:\n"
        "- <edge case to handle>\n"
        "- <edge case to handle>\n\n"
        "IMPORTANT: Only plan changes to the specific function(s) relevant to the task. "
        "If the file contains other functions or classes, they must NOT be modified or removed.\n\n"
        "Be specific about data structures, loop bounds, and conditions. Max 6 steps."
    )

    try:
        result = llm.invoke(prompt).strip()
        return _parse_plan(result)
    except Exception as e:
        logger.warning("[PLANNER] Planning failed: %s", e)
        return {"steps": [instruction], "approach": "", "edge_cases": []}


def _parse_plan(raw: str) -> Dict:
    """Parse the planner LLM output into structured form."""
    approach = ""
    steps = []
    edge_cases = []
    section = None

    for line in raw.split("\n"):
        line_stripped = line.strip()
        upper = line_stripped.upper()

        if upper.startswith("APPROACH:"):
            approach = line_stripped[len("APPROACH:"):].strip()
            section = "approach"
        elif upper.startswith("STEPS:"):
            section = "steps"
        elif upper.startswith("EDGE_CASES:") or upper.startswith("EDGE CASES:"):
            section = "edge_cases"
        elif section == "steps" and line_stripped:
            # Strip leading number/bullet: "1. xxx" → "xxx"
            import re
            clean = re.sub(r'^\d+[\.\)]\s*', '', line_stripped)
            if clean:
                steps.append(clean)
        elif section == "edge_cases" and line_stripped:
            clean = line_stripped.lstrip("-•* ").strip()
            if clean:
                edge_cases.append(clean)

    return {
        "steps": steps if steps else [raw[:500]],
        "approach": approach,
        "edge_cases": edge_cases,
    }


def format_plan_for_prompt(plan: Dict) -> str:
    """Format a plan dict into a string suitable for injection into an edit prompt."""
    parts = []
    if plan.get("approach"):
        parts.append(f"PLANNED APPROACH: {plan['approach']}")
    if plan.get("steps"):
        parts.append("IMPLEMENTATION STEPS:")
        for i, step in enumerate(plan["steps"], 1):
            parts.append(f"  {i}. {step}")
    if plan.get("edge_cases"):
        parts.append("EDGE CASES TO HANDLE:")
        for ec in plan["edge_cases"]:
            parts.append(f"  - {ec}")
    return "\n".join(parts) + "\n"


# ── Critic ───────────────────────────────────────────────────────────────────

def critique_code(
    instruction: str,
    original_source: str,
    new_source: str,
    ext: str,
    test_results: Optional[List[Dict]] = None,
    rag_context: str = "",
    llm: Optional[OllamaLLM] = None,
) -> Dict:
    """
    Critique the agent's output: verify correctness against requirements and docs.

    Returns:
        {
            "verdict": "PASS" | "FAIL" | "PARTIAL",
            "issues": ["issue 1", "issue 2", ...],
            "suggestions": ["suggestion 1", ...],
            "confidence": float (0.0 - 1.0),
        }
    """
    if llm is None:
        llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)

    lang = LANG_FENCE.get(ext, ext.lstrip('.'))

    test_section = ""
    if test_results:
        test_lines = []
        for i, r in enumerate(test_results, 1):
            status = "✓ PASS" if r.get("passed") else "✗ FAIL"
            got = r.get("actual", r.get("error", "N/A"))
            test_lines.append(f"  Test {i}: {status} — input={r['input']} expected={r['expected']} got={got}")
        test_section = "TEST RESULTS:\n" + "\n".join(test_lines) + "\n\n"

    rag_section = ""
    if rag_context:
        rag_section = f"REFERENCE DOCS (verify against these):\n{rag_context[:1500]}\n\n"

    prompt = (
        f"You are a code reviewer verifying an agent's work.\n\n"
        f"REQUIREMENT: {instruction}\n\n"
        f"ORIGINAL CODE:\n```{lang}\n{original_source[:2000]}\n```\n\n"
        f"MODIFIED CODE:\n```{lang}\n{new_source[:2000]}\n```\n\n"
        f"{test_section}"
        f"{rag_section}"
        f"Evaluate the modified code:\n"
        f"1. Does it correctly satisfy the requirement?\n"
        f"2. Does the logic match what the reference docs describe (if available)?\n"
        f"3. Are there any bugs, off-by-one errors, or missed edge cases?\n"
        f"4. Does it handle the test cases correctly?\n\n"
        f"Output EXACTLY this format:\n"
        f"VERDICT: PASS | FAIL | PARTIAL\n"
        f"CONFIDENCE: 0.0 to 1.0\n"
        f"ISSUES:\n- <issue> (or 'None')\n"
        f"SUGGESTIONS:\n- <suggestion> (or 'None')\n"
    )

    try:
        result = llm.invoke(prompt).strip()
        return _parse_critique(result)
    except Exception as e:
        logger.warning("[CRITIC] Critique failed: %s", e)
        return {"verdict": "PASS", "issues": [], "suggestions": [], "confidence": 0.5}


def _parse_critique(raw: str) -> Dict:
    """Parse the critic LLM output into structured form."""
    verdict = "PASS"
    confidence = 0.5
    issues = []
    suggestions = []
    section = None

    for line in raw.split("\n"):
        line_stripped = line.strip()
        upper = line_stripped.upper()

        if upper.startswith("VERDICT:"):
            v = line_stripped[len("VERDICT:"):].strip().upper()
            if "FAIL" in v:
                verdict = "FAIL"
            elif "PARTIAL" in v:
                verdict = "PARTIAL"
            else:
                verdict = "PASS"
        elif upper.startswith("CONFIDENCE:"):
            try:
                confidence = float(line_stripped[len("CONFIDENCE:"):].strip())
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass
        elif upper.startswith("ISSUES:"):
            section = "issues"
        elif upper.startswith("SUGGESTIONS:"):
            section = "suggestions"
        elif section == "issues" and line_stripped:
            clean = line_stripped.lstrip("-•* ").strip()
            if clean.lower() != "none" and clean:
                issues.append(clean)
        elif section == "suggestions" and line_stripped:
            clean = line_stripped.lstrip("-•* ").strip()
            if clean.lower() != "none" and clean:
                suggestions.append(clean)

    return {
        "verdict": verdict,
        "issues": issues,
        "suggestions": suggestions,
        "confidence": confidence,
    }


def build_critic_feedback_note(critique: Dict) -> str:
    """Build a prompt note from critic feedback for a retry loop."""
    if critique["verdict"] == "PASS" and not critique["issues"]:
        return ""

    parts = [f"CRITIC VERDICT: {critique['verdict']} (confidence: {critique['confidence']:.1f})"]
    if critique["issues"]:
        parts.append("ISSUES FOUND:")
        for issue in critique["issues"]:
            parts.append(f"  - {issue}")
    if critique["suggestions"]:
        parts.append("SUGGESTIONS:")
        for s in critique["suggestions"]:
            parts.append(f"  - {s}")
    parts.append("\nFix the issues identified above. Output a SEARCH/REPLACE block.")
    return "\n".join(parts)
