"""
Tutor module — interactive problem-solving engine.

Generates coding/concept problems, evaluates user answers (MCQ or free text),
provides progressive hints, and runs user code against test cases.
"""

import json
import re
import uuid
import random
from typing import Optional

from langchain_ollama import OllamaLLM
from brain.config import LLM_MODEL, LLM_TEMPERATURE

# In-memory store of active tutor sessions  {session_id: TutorState}
_tutor_sessions: dict[str, dict] = {}

# Track recently generated questions per topic to avoid repetition
# Key: (topic, difficulty, style), Value: list of question summaries
_question_history: dict[tuple, list] = {}


def _llm(temperature: float = 0.3) -> OllamaLLM:
    return OllamaLLM(model=LLM_MODEL, temperature=temperature)


def _parse_json_from_llm(text: str) -> dict:
    """Best-effort extraction of a JSON object from LLM output.

    Handles common LLM quirks: markdown fences, literal newlines inside string
    values, triple-quoted strings, and multiple JSON blocks.
    """
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())

    # Try direct parse first (fast path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 1: Find the outermost {...} and try parsing just that
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start != -1 and brace_end > brace_start:
        candidate = text[brace_start:brace_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Escape literal newlines inside JSON string values.
        # Walk character by character to find unescaped newlines within "..."
        fixed = _fix_json_newlines(candidate)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # Strategy 3: Replace triple-quoted strings (""" ... """) with single-quoted
    triple_fixed = re.sub(
        r'"""(.*?)"""',
        lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\r', '').replace('"', '\\"') + '"',
        text,
        flags=re.DOTALL,
    )
    try:
        return json.loads(triple_fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Aggressive line-by-line newline escaping inside the JSON block
    if brace_start != -1 and brace_end > brace_start:
        aggressive = text[brace_start:brace_end + 1]
        aggressive = _fix_json_newlines(aggressive)
        # Also handle trailing commas before } or ]
        aggressive = re.sub(r',\s*([}\]])', r'\1', aggressive)
        try:
            return json.loads(aggressive)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not parse JSON from LLM output", text, 0)


def _fix_json_newlines(text: str) -> str:
    """Escape literal newlines that appear inside JSON string values."""
    result = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == '\\':
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string and ch == '\n':
            result.append('\\n')
            continue
        if in_string and ch == '\r':
            continue
        if in_string and ch == '\t':
            result.append('\\t')
            continue
        result.append(ch)
    return ''.join(result)


# ── Lesson Generation ──────────────────────────────────────────────────────────

def generate_lesson(
    topic: str,
    difficulty: str = "medium",
    language: str = "python",
) -> dict:
    """Generate a lesson covering the fundamentals of a topic.

    Returns dict with keys:
      title, explanation, rules, example_code, example_explanation
    """
    llm = _llm(temperature=0.4)

    prompt = f"""You are an expert programming tutor. Generate a concise lesson about "{topic}" in {language} at {difficulty} difficulty level.

The lesson should cover the key fundamentals that a student needs to understand before solving problems on this topic.

Return ONLY a JSON object with this EXACT structure:
{{
  "title": "Topic Title (e.g. Arrays in C++)",
  "explanation": "A clear 2-4 paragraph explanation of the core concept, how it works, and why it matters. Be specific to {language}.",
  "rules": [
    "Rule 1: specific rule or behavior the student must understand",
    "Rule 2: another important rule",
    "Rule 3: edge case or common pitfall"
  ],
  "example_code": "A complete, runnable code example demonstrating the concept. Include comments.",
  "example_explanation": "Line-by-line or section-by-section walkthrough of what the example code does",
  "key_terms": ["term1", "term2", "term3"]
}}

Make the lesson practical and concrete. Use {language}-specific syntax and idioms. The example code should be realistic and demonstrate the concept clearly."""

    raw = llm.invoke(prompt)
    try:
        lesson = _parse_json_from_llm(raw)
    except (json.JSONDecodeError, ValueError):
        lesson = {
            "title": f"{topic} in {language}",
            "explanation": raw.strip(),
            "rules": [],
            "example_code": "",
            "example_explanation": "",
            "key_terms": [],
        }
    return lesson


# ── Problem Generation ─────────────────────────────────────────────────────────

def generate_problem(
    topic: str,
    difficulty: str = "medium",
    language: str = "python",
    style: str = "mcq",  # "mcq" | "free_text" | "code"
) -> dict:
    """Generate a practice problem with an accompanying lesson.

    Returns dict with keys:
      session_id, style, question, options (if mcq), correct_answer,
      test_cases (if code), hints[], lesson, code_snippet (if mcq)
    """
    # Step 1: Generate the lesson first
    lesson = generate_lesson(topic, difficulty, language)

    # Step 2: Build variation context from question history
    history_key = (topic.lower(), difficulty, style)
    prev_questions = _question_history.get(history_key, [])

    # Build avoidance instruction so the LLM generates a different question
    variation_note = ""
    if prev_questions:
        recent = prev_questions[-5:]  # last 5 questions on this topic
        avoid_list = "\n".join(f"  - {q}" for q in recent)
        variation_note = (
            f"\n\nIMPORTANT: The student has already seen these questions on this topic. "
            f"Generate a COMPLETELY DIFFERENT question that tests a different aspect:\n{avoid_list}\n\n"
            f"Use a different code pattern, different edge case, or different sub-topic within {topic}."
        )

    # Add randomized angle to encourage variety
    angles = [
        "Focus on a common bug or pitfall.",
        "Focus on edge cases and boundary conditions.",
        "Focus on performance and time complexity.",
        "Focus on how the data structure works internally.",
        "Focus on comparing two approaches to the same problem.",
        "Focus on debugging — include a subtle bug in the code.",
        "Focus on output prediction — what does this code print?",
        "Focus on memory and space complexity.",
    ]
    random_angle = random.choice(angles)

    # Step 3: Generate the problem
    llm = _llm(temperature=0.6)  # slightly higher temp for more variety

    if style == "mcq":
        prompt = f"""Generate a multiple-choice question about {topic} ({difficulty} difficulty) in {language}.

Angle: {random_angle}

IMPORTANT: The question MUST include a code snippet that the student needs to analyze.
The code snippet should be actual {language} code that demonstrates a concept, bug, or behavior related to {topic}.
The question should ask what happens when the code runs, what's wrong with it, or what output it produces.
{variation_note}
Return ONLY a JSON object with this EXACT structure:
{{
  "code_snippet": "The actual {language} code snippet that the student must read and analyze. Multiple lines, properly formatted.",
  "question": "Based on the code above, what happens when... / what is the output of... / which statement is correct about...",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "correct_answer": "A",
  "explanation": "Detailed explanation of why the correct answer is correct, referencing specific lines of the code",
  "hints": ["First hint about what to look for in the code", "Second stronger hint referencing a specific line"]
}}

The code_snippet MUST be non-trivial, compilable/runnable {language} code. Do NOT reference a code snippet without providing one."""
    elif style == "free_text":
        prompt = f"""Generate a short-answer conceptual question about {topic} ({difficulty} difficulty) for {language}.
The student should explain a concept in their own words.

Angle: {random_angle}
{variation_note}
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

Angle: {random_angle}
{variation_note}
Provide a clear problem description with:
- What the function should do
- The function signature
- Rules and constraints
- An example with input/output
- Implementation hints

Return ONLY a JSON object with this EXACT structure:
{{
  "question": "Detailed problem description including function signature, rules, example usage, and implementation hints. Use newlines for formatting.",
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
        "lesson": lesson,
        "hints_given": 0,
        "attempts": 0,
        "solved": False,
    }
    _tutor_sessions[session_id] = state

    # Track question in history to avoid repetition
    question_summary = problem.get("question", "")[:100]
    if question_summary:
        _question_history.setdefault(history_key, []).append(question_summary)
        # Keep only last 20 questions per key to avoid unbounded growth
        if len(_question_history[history_key]) > 20:
            _question_history[history_key] = _question_history[history_key][-20:]

    # Build response (don't leak correct_answer to frontend)
    resp: dict = {
        "session_id": session_id,
        "style": style,
        "question": problem.get("question", ""),
        "language": language,
        "lesson": lesson,
    }
    if style == "mcq":
        resp["options"] = problem.get("options", [])
        resp["code_snippet"] = problem.get("code_snippet", "")
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
        "code_snippet": state["problem"].get("code_snippet"),
        "test_cases": state["problem"].get("test_cases"),
        "function_name": state["problem"].get("function_name"),
        "lesson": state.get("lesson"),
        "hints_given": state["hints_given"],
        "attempts": state["attempts"],
        "solved": state["solved"],
    }


# ── Agent Learnings Bridge ─────────────────────────────────────────────────────
# Stores explanations generated by the code agent so the tutor can reference them.

_agent_learnings: list[dict] = []  # [{topic, language, explanation, timestamp}]
MAX_LEARNINGS = 50


def store_agent_learning(explanation: str, topic: str = "", language: str = "", file_path: str = "") -> None:
    """Store a code agent explanation for use in tutor mode."""
    if not explanation or len(explanation.strip()) < 20:
        return
    from datetime import datetime
    _agent_learnings.append({
        "topic": topic,
        "language": language,
        "explanation": explanation,
        "file_path": file_path,
        "timestamp": datetime.now().isoformat(),
    })
    # Cap the list
    if len(_agent_learnings) > MAX_LEARNINGS:
        _agent_learnings[:] = _agent_learnings[-MAX_LEARNINGS:]


def get_agent_learnings(topic: str = "", language: str = "", limit: int = 5) -> list:
    """Retrieve recent agent learnings, optionally filtered by topic/language."""
    results = _agent_learnings[:]
    if topic:
        topic_lower = topic.lower()
        results = [l for l in results if topic_lower in l.get("topic", "").lower() or topic_lower in l.get("explanation", "").lower()]
    if language:
        lang_lower = language.lower()
        results = [l for l in results if lang_lower in l.get("language", "").lower()]
    return results[-limit:]
