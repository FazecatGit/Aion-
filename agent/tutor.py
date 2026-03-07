"""
Tutor module — interactive problem-solving engine.

Generates coding/concept problems, evaluates user answers (MCQ or free text),
provides progressive hints, and runs user code against test cases.
"""

import json
import re
import uuid
from typing import Optional

from langchain_ollama import OllamaLLM
from brain.config import LLM_MODEL, LLM_TEMPERATURE

# In-memory store of active tutor sessions  {session_id: TutorState}
_tutor_sessions: dict[str, dict] = {}


def _llm(temperature: float = 0.3) -> OllamaLLM:
    return OllamaLLM(model=LLM_MODEL, temperature=temperature)


def _parse_json_from_llm(text: str) -> dict:
    """Best-effort extraction of a JSON object from LLM output."""
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return json.loads(text)


# ── Problem Generation ─────────────────────────────────────────────────────────

def generate_problem(
    topic: str,
    difficulty: str = "medium",
    language: str = "python",
    style: str = "mcq",  # "mcq" | "free_text" | "code"
) -> dict:
    """Generate a single practice problem.

    Returns dict with keys:
      session_id, style, question, options (if mcq), correct_answer,
      test_cases (if code), hints[]
    """
    llm = _llm(temperature=0.5)

    if style == "mcq":
        prompt = f"""Generate a multiple-choice question about {topic} ({difficulty} difficulty).
The question should test understanding of {language} programming concepts.

Return ONLY a JSON object with this EXACT structure:
{{
  "question": "The question text",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "correct_answer": "A",
  "explanation": "Why the correct answer is correct",
  "hints": ["First hint", "Second stronger hint"]
}}"""
    elif style == "free_text":
        prompt = f"""Generate a short-answer conceptual question about {topic} ({difficulty} difficulty) for {language}.
The student should explain a concept in their own words.

Return ONLY a JSON object with this EXACT structure:
{{
  "question": "The question text",
  "correct_answer": "A model answer covering the key points",
  "key_points": ["point 1", "point 2", "point 3"],
  "hints": ["First hint", "Second stronger hint"]
}}"""
    else:  # code
        prompt = f"""Generate a coding problem about {topic} ({difficulty} difficulty) in {language}.
The problem should require the student to write a function.

Return ONLY a JSON object with this EXACT structure:
{{
  "question": "Problem description including function signature",
  "function_name": "the_function_name",
  "test_cases": [
    {{"input": "input_value_1", "expected": "expected_output_1"}},
    {{"input": "input_value_2", "expected": "expected_output_2"}},
    {{"input": "input_value_3", "expected": "expected_output_3"}}
  ],
  "correct_answer": "A reference solution",
  "hints": ["First hint about approach", "Second hint with more detail", "Third hint nearly giving it away"]
}}"""

    raw = llm.invoke(prompt)
    try:
        problem = _parse_json_from_llm(raw)
    except (json.JSONDecodeError, ValueError):
        # Fallback: return the raw text as a free-text question
        problem = {
            "question": raw.strip(),
            "correct_answer": "",
            "hints": [],
        }

    session_id = str(uuid.uuid4())
    state = {
        "session_id": session_id,
        "style": style,
        "topic": topic,
        "difficulty": difficulty,
        "language": language,
        "problem": problem,
        "hints_given": 0,
        "attempts": 0,
        "solved": False,
    }
    _tutor_sessions[session_id] = state

    # Build response (don't leak correct_answer to frontend)
    resp: dict = {
        "session_id": session_id,
        "style": style,
        "question": problem.get("question", ""),
        "language": language,
    }
    if style == "mcq":
        resp["options"] = problem.get("options", [])
    if style == "code":
        resp["test_cases"] = problem.get("test_cases", [])
        resp["function_name"] = problem.get("function_name", "")
    return resp


# ── Answer Checking ────────────────────────────────────────────────────────────

def check_answer(session_id: str, user_answer: str) -> dict:
    """Evaluate the user's answer. Returns {correct, feedback, solved}."""
    state = _tutor_sessions.get(session_id)
    if not state:
        return {"error": "Session not found"}

    state["attempts"] += 1
    problem = state["problem"]
    style = state["style"]

    if style == "mcq":
        correct_letter = problem.get("correct_answer", "").strip().upper()
        user_letter = user_answer.strip().upper()
        # Accept "A", "A)", "A) ..." formats
        if user_letter and user_letter[0] == correct_letter[0] if correct_letter else False:
            state["solved"] = True
            return {
                "correct": True,
                "feedback": problem.get("explanation", "Correct!"),
                "solved": True,
                "attempts": state["attempts"],
            }
        else:
            return {
                "correct": False,
                "feedback": "That's not right. Think about it again, or ask for a hint.",
                "solved": False,
                "attempts": state["attempts"],
            }

    elif style == "free_text":
        # Use LLM to judge the free-text answer
        key_points = problem.get("key_points", [])
        model_answer = problem.get("correct_answer", "")
        llm = _llm(temperature=0.1)
        prompt = f"""You are grading a student's answer to a programming question.

Question: {problem.get('question', '')}

Model answer: {model_answer}
Key points that should be covered: {json.dumps(key_points)}

Student's answer: {user_answer}

Evaluate the student's answer. Return ONLY a JSON object:
{{
  "correct": true/false,
  "score": 0-100,
  "feedback": "Specific feedback on what they got right and what they missed",
  "missing_points": ["any key points they did not cover"]
}}"""
        raw = llm.invoke(prompt)
        try:
            result = _parse_json_from_llm(raw)
        except (json.JSONDecodeError, ValueError):
            result = {"correct": False, "score": 0, "feedback": raw.strip(), "missing_points": []}

        is_correct = result.get("correct", False) or result.get("score", 0) >= 70
        if is_correct:
            state["solved"] = True
        return {
            "correct": is_correct,
            "feedback": result.get("feedback", ""),
            "score": result.get("score", 0),
            "missing_points": result.get("missing_points", []),
            "solved": is_correct,
            "attempts": state["attempts"],
        }

    else:  # code — just tell the user to use the run endpoint
        return {
            "correct": False,
            "feedback": "Use the Run Code button to test your solution against the test cases.",
            "solved": False,
            "attempts": state["attempts"],
        }


# ── Code Execution ─────────────────────────────────────────────────────────────

def run_tutor_code(session_id: str, user_code: str) -> dict:
    """Run user's code against the problem's test cases.
    Returns {results: [{input, expected, actual, passed}], all_passed, solved}"""
    state = _tutor_sessions.get(session_id)
    if not state:
        return {"error": "Session not found"}

    problem = state["problem"]
    test_cases = problem.get("test_cases", [])
    language = state.get("language", "python")
    func_name = problem.get("function_name", "solution")

    if not test_cases:
        return {"error": "No test cases for this problem"}

    from agent.test_runner import run_tests

    # Build a file path hint so the test runner picks the right language
    ext_map = {"python": ".py", "go": ".go", "cpp": ".cpp", "c": ".c",
               "javascript": ".js", "typescript": ".ts", "java": ".java", "rust": ".rs"}
    fake_path = f"tutor_solution{ext_map.get(language, '.py')}"

    results = run_tests(user_code, fake_path, test_cases)
    all_passed = all(r["passed"] for r in results)
    if all_passed:
        state["solved"] = True
        state["attempts"] += 1

    return {
        "results": results,
        "all_passed": all_passed,
        "solved": all_passed,
        "attempts": state["attempts"],
    }


# ── Hints ──────────────────────────────────────────────────────────────────────

def get_hint(session_id: str) -> dict:
    """Return the next progressive hint for the current problem."""
    state = _tutor_sessions.get(session_id)
    if not state:
        return {"error": "Session not found"}

    problem = state["problem"]
    hints = problem.get("hints", [])
    idx = state["hints_given"]

    if idx < len(hints):
        hint = hints[idx]
        state["hints_given"] = idx + 1
        return {
            "hint": hint,
            "hint_number": idx + 1,
            "total_hints": len(hints),
        }
    else:
        # No more pre-generated hints — generate one dynamically
        llm = _llm(temperature=0.3)
        prompt = f"""The student is stuck on this problem:

{problem.get('question', '')}

They have used all {len(hints)} hints. Give them one more helpful hint that gets
them closer to the answer without giving it away completely.
Return just the hint text, nothing else."""
        new_hint = llm.invoke(prompt).strip()
        state["hints_given"] += 1
        return {
            "hint": new_hint,
            "hint_number": state["hints_given"],
            "total_hints": "∞",
        }


# ── Session management ─────────────────────────────────────────────────────────

def get_tutor_state(session_id: str) -> Optional[dict]:
    """Return current tutor session state (without leaking answer)."""
    state = _tutor_sessions.get(session_id)
    if not state:
        return None
    return {
        "session_id": state["session_id"],
        "style": state["style"],
        "topic": state["topic"],
        "difficulty": state["difficulty"],
        "language": state["language"],
        "question": state["problem"].get("question", ""),
        "options": state["problem"].get("options"),
        "test_cases": state["problem"].get("test_cases"),
        "function_name": state["problem"].get("function_name"),
        "hints_given": state["hints_given"],
        "attempts": state["attempts"],
        "solved": state["solved"],
    }
