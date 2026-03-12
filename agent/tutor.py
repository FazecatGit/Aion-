"""
Tutor module — interactive problem-solving engine.

Generates coding/concept problems, evaluates user answers (MCQ or free text),
provides progressive hints, and runs user code against test cases.
"""

import json
import re
import uuid
import random
import logging
from datetime import datetime
from typing import Optional

from langchain_ollama import OllamaLLM
from brain.config import LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger("tutor")

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


# ═══════════════════════════════════════════════════════════════════════════════
# LESSON GENERATION
# ═══════════════════════════════════════════════════════════════════════════════────

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

    # Pull relevant agent learnings so the lesson incorporates real debugging insights
    learnings = get_agent_learnings(topic=topic, language=language, limit=3)
    learnings_block = ""
    if learnings:
        snippets = []
        for l in learnings:
            snippets.append(f"- {l.get('explanation', '')[:500]}")
        learnings_block = (
            "\n\nThe student's code agent previously generated these explanations on related topics. "
            "Incorporate relevant insights, patterns, or edge cases from them into the lesson:\n"
            + "\n".join(snippets) + "\n"
        )

    prompt = f"""You are an expert programming tutor. Generate a concise lesson about "{topic}" in {language} at {difficulty} difficulty level.

The lesson should cover the key fundamentals that a student needs to understand before solving problems on this topic.
{learnings_block}
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
        # Aggressive retry: strip everything outside outermost braces, fix common issues
        fallback_parsed = None
        try:
            text = raw.strip()
            # Remove markdown fences anywhere
            text = re.sub(r'```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```', '', text)
            brace_start = text.find('{')
            brace_end = text.rfind('}')
            if brace_start != -1 and brace_end > brace_start:
                candidate = text[brace_start:brace_end + 1]
                # Escape all literal newlines inside string values
                fixed = _fix_json_newlines(candidate)
                # Remove trailing commas
                fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
                fallback_parsed = json.loads(fixed)
        except Exception:
            pass

        if fallback_parsed and isinstance(fallback_parsed, dict) and fallback_parsed.get("explanation"):
            lesson = fallback_parsed
        else:
            lesson = {
                "title": f"{topic} in {language}",
                "explanation": raw.strip(),
                "rules": [],
                "example_code": "",
                "example_explanation": "",
            "key_terms": [],
        }
    # Sanitize lesson fields — unwrap any JSON strings
    if isinstance(lesson.get("explanation"), str):
        lesson["explanation"] = _try_unwrap_json_string(lesson["explanation"])
    return lesson


# ═══════════════════════════════════════════════════════════════════════════════
# PROBLEM SANITISATION — prevent JSON slop from reaching the frontend
# ═══════════════════════════════════════════════════════════════════════════════────

def _try_unwrap_json_string(value: str) -> str:
    """If a string value is itself a JSON object, extract the meaningful text from it."""
    v = value.strip()
    if not (v.startswith('{') and v.endswith('}')):
        return value
    try:
        parsed = json.loads(v)
        if isinstance(parsed, dict):
            # Try common keys that hold the real text
            for key in ('question', 'text', 'content', 'explanation', 'code_snippet', 'code'):
                if key in parsed and isinstance(parsed[key], str):
                    return parsed[key]
            # Fall back to joining all string values
            parts = [str(val) for val in parsed.values() if isinstance(val, str) and len(str(val)) > 10]
            if parts:
                return "\n".join(parts)
    except (json.JSONDecodeError, ValueError):
        pass
    return value


def _sanitize_problem(problem: dict, style: str) -> dict:
    """Clean up a parsed problem dict so no field contains raw JSON slop."""
    # Clean question
    q = problem.get("question", "")
    if isinstance(q, str):
        problem["question"] = _try_unwrap_json_string(q)

    # Clean code_snippet
    cs = problem.get("code_snippet", "")
    if isinstance(cs, str):
        problem["code_snippet"] = _try_unwrap_json_string(cs)

    # Clean explanation
    exp = problem.get("explanation", "")
    if isinstance(exp, str):
        problem["explanation"] = _try_unwrap_json_string(exp)

    # Clean options — each must be a plain string like "A) ..."
    if style == "mcq":
        raw_opts = problem.get("options", [])
        if isinstance(raw_opts, list):
            cleaned_opts = []
            for i, opt in enumerate(raw_opts):
                if isinstance(opt, dict):
                    # LLM returned {label: ..., text: ...} instead of a string
                    label = chr(65 + i)  # A, B, C, D
                    text = opt.get("text", opt.get("label", str(opt)))
                    cleaned_opts.append(f"{label}) {text}")
                elif isinstance(opt, str):
                    cleaned_opts.append(opt)
                else:
                    cleaned_opts.append(str(opt))
            problem["options"] = cleaned_opts

    # Clean hints
    hints = problem.get("hints", [])
    if isinstance(hints, list):
        problem["hints"] = [str(h) for h in hints if h]
    elif isinstance(hints, str):
        problem["hints"] = [hints]

    # Ensure correct_answer is a simple letter for MCQ
    if style == "mcq":
        ca = problem.get("correct_answer", "")
        if isinstance(ca, str) and ca:
            # Normalize to just the letter
            ca = ca.strip().upper()
            if ca and ca[0] in "ABCD":
                problem["correct_answer"] = ca[0]

    return problem


# ═══════════════════════════════════════════════════════════════════════════════
# PROBLEM GENERATION
# ═══════════════════════════════════════════════════════════════════════════════────

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

    # ── Validate and clean the parsed problem fields ──────────────────────────
    # Prevents JSON slop from reaching the frontend.
    problem = _sanitize_problem(problem, style)

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


# ═══════════════════════════════════════════════════════════════════════════════
# ANSWER CHECKING
# ═══════════════════════════════════════════════════════════════════════════════────

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


# ═══════════════════════════════════════════════════════════════════════════════
# CODE EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════────

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


# ═══════════════════════════════════════════════════════════════════════════════
# HINTS
# ═══════════════════════════════════════════════════════════════════════════════────

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


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════────

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


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT LEARNINGS BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════────
# Stores explanations generated by the code agent so the tutor can reference them.

_agent_learnings: list[dict] = []  # [{topic, language, explanation, timestamp}]
MAX_LEARNINGS = 50


def store_agent_learning(explanation: str, topic: str = "", language: str = "", file_path: str = "") -> None:
    """Store a code agent explanation for use in tutor mode."""
    if not explanation or len(explanation.strip()) < 20:
        return
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
        # Keyword-based matching: split topic into words, match if >=2 keywords hit
        topic_clean = topic.replace('…', '').replace('...', '').strip().lower()
        keywords = [w for w in re.split(r'\W+', topic_clean) if len(w) >= 3]
        if keywords:
            def _matches(learning: dict) -> bool:
                haystack = (learning.get("topic", "") + " " + learning.get("explanation", "")).lower()
                hits = sum(1 for kw in keywords if kw in haystack)
                # Match if at least 2 keywords hit, or 1 if there's only 1 keyword
                return hits >= min(2, len(keywords))
            results = [l for l in results if _matches(l)]
    if language:
        lang_lower = language.lower()
        results = [l for l in results if lang_lower in l.get("language", "").lower()]
    return results[-limit:]


# ═══════════════════════════════════════════════════════════════════════════════
# MATH TOPICS — for detecting when to use math tutor mode
# ═══════════════════════════════════════════════════════════════════════════════

_MATH_TOPIC_KEYWORDS = {
    'calculus', 'derivative', 'integration', 'integral', 'differentiation',
    'linear algebra', 'matrix', 'matrices', 'vector', 'eigenvalue',
    'statistics', 'probability', 'discrete math', 'geometry', 'trigonometry',
    'optimization', 'number theory', 'combinatorics', 'polynomial',
    'quadratic', 'logarithm', 'exponential', 'limit', 'continuity',
    'taylor series', 'fourier', 'laplace', 'differential equation',
    'partial derivative', 'gradient', 'divergence', 'curl',
    'determinant', 'inverse matrix', 'systems of equations',
    'permutation', 'combination', 'binomial', 'factoring',
    'arithmetic', 'modular arithmetic', 'prime', 'gcd', 'lcm',
    'complex number', 'imaginary', 'sin', 'cos', 'tan', 'pythagorean',
    'area', 'volume', 'perimeter', 'mean', 'median', 'variance',
    'standard deviation', 'regression', 'hypothesis', 'bayes',
    'math', 'equation', 'simplify', 'evaluate', 'prove',
}


def is_math_topic(topic: str) -> bool:
    """Return True if the topic string relates to mathematics."""
    t = topic.lower()
    return any(kw in t for kw in _MATH_TOPIC_KEYWORDS)


# ═══════════════════════════════════════════════════════════════════════════════
# MATH PROBLEM GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_math_problem(
    topic: str,
    difficulty: str = "medium",
    style: str = "solve",  # "solve" | "mcq" | "proof"
) -> dict:
    """Generate a math practice problem with step-by-step solution.

    Returns dict with keys:
      session_id, style, question, correct_answer, steps, hints, lesson
    """
    logger.info("[MATH] Generating %s problem — topic=%s, difficulty=%s", style, topic, difficulty)
    # Generate a math lesson first
    logger.info("[MATH] Step 1/3: Generating lesson for '%s'...", topic)
    lesson = _generate_math_lesson(topic, difficulty)
    logger.info("[MATH] Step 1/3: Lesson generated ✓")

    # Build variation context
    history_key = (f"math:{topic.lower()}", difficulty, style)
    prev_questions = _question_history.get(history_key, [])
    variation_note = ""
    if prev_questions:
        recent = prev_questions[-5:]
        avoid_list = "\n".join(f"  - {q}" for q in recent)
        variation_note = (
            f"\n\nIMPORTANT: The student has already seen these problems. "
            f"Generate a COMPLETELY DIFFERENT problem:\n{avoid_list}\n"
        )

    angles = [
        "Focus on a common mistake students make.",
        "Use a real-world application context.",
        "Test understanding of edge cases.",
        "Require multiple steps to solve.",
        "Combine two related concepts.",
        "Use larger or unusual numbers.",
    ]
    random_angle = random.choice(angles)

    logger.info("[MATH] Step 2/3: Invoking LLM to generate %s problem...", style)
    llm = _llm(temperature=0.6)

    if style == "mcq":
        prompt = f"""Generate a multiple-choice math question about {topic} at {difficulty} difficulty.

Angle: {random_angle}
{variation_note}
Return ONLY a JSON object:
{{
  "question": "The math question with any necessary equations (use plain text math notation)",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "correct_answer": "A",
  "steps": [
    "Step 1: description of first step with computation",
    "Step 2: description of next step",
    "Step 3: final step arriving at the answer"
  ],
  "explanation": "Why the correct answer is right and common mistakes",
  "hints": ["First hint", "Second hint"]
}}"""
    elif style == "proof":
        prompt = f"""Generate a mathematical proof problem about {topic} at {difficulty} difficulty.

Angle: {random_angle}
{variation_note}
Return ONLY a JSON object:
{{
  "question": "Prove that ... (clear statement of what to prove)",
  "correct_answer": "A model proof with clear logical steps",
  "steps": [
    "Step 1: State what we need to prove",
    "Step 2: Key insight or lemma",
    "Step 3: Main argument",
    "Step 4: Conclusion"
  ],
  "key_points": ["Key concept 1", "Key concept 2"],
  "hints": ["First hint about approach", "Second hint with more detail"]
}}"""
    else:  # solve
        prompt = f"""Generate a math problem to solve about {topic} at {difficulty} difficulty.

Angle: {random_angle}
{variation_note}
The problem should require step-by-step computation. Provide specific numbers.

Return ONLY a JSON object:
{{
  "question": "The problem statement with specific values (use plain text math notation like x^2 for x squared, sqrt(x) for square root)",
  "correct_answer": "The final numerical or symbolic answer",
  "steps": [
    "Step 1: Identify what we need to find and write the equation",
    "Step 2: Apply the relevant formula or technique",
    "Step 3: Simplify or compute",
    "Step 4: State the final answer"
  ],
  "hints": ["First hint about what technique to use", "Second hint with more detail", "Third hint nearly giving the approach"],
  "related_formulas": ["Formula 1: description", "Formula 2: description"]
}}"""

    raw = llm.invoke(prompt)
    logger.info("[MATH] Step 2/3: LLM response received ✓")
    try:
        problem = _parse_json_from_llm(raw)
        logger.info("[MATH] Step 3/3: Problem parsed successfully ✓")
    except (json.JSONDecodeError, ValueError):
        logger.warning("[MATH] Step 3/3: JSON parse failed — using raw text as question")
        problem = {
            "question": raw.strip(),
            "correct_answer": "",
            "steps": [],
            "hints": [],
        }

    # Sanitize
    problem = _sanitize_problem(problem, style)

    session_id = str(uuid.uuid4())
    logger.info("[MATH] Problem ready — session=%s", session_id)
    state = {
        "session_id": session_id,
        "style": style,
        "topic": topic,
        "difficulty": difficulty,
        "language": "math",
        "problem": problem,
        "lesson": lesson,
        "hints_given": 0,
        "attempts": 0,
        "solved": False,
        "is_math": True,
    }
    _tutor_sessions[session_id] = state

    # Track history
    question_summary = problem.get("question", "")[:100]
    if question_summary:
        _question_history.setdefault(history_key, []).append(question_summary)
        if len(_question_history[history_key]) > 20:
            _question_history[history_key] = _question_history[history_key][-20:]

    resp: dict = {
        "session_id": session_id,
        "style": style,
        "question": problem.get("question", ""),
        "language": "math",
        "lesson": lesson,
        "is_math": True,
    }
    if style == "mcq":
        resp["options"] = problem.get("options", [])
    if style == "solve":
        resp["related_formulas"] = problem.get("related_formulas", [])
    return resp


def _generate_math_lesson(topic: str, difficulty: str = "medium") -> dict:
    """Generate a math lesson (not code-based)."""
    logger.info("[MATH] Generating lesson for '%s' (%s)...", topic, difficulty)
    llm = _llm(temperature=0.4)
    prompt = f"""You are an expert math tutor. Generate a concise lesson about "{topic}" at {difficulty} difficulty.

Return ONLY a JSON object:
{{
  "title": "Topic Title (e.g. Derivatives - Chain Rule)",
  "explanation": "Clear 2-4 paragraph explanation of the concept, its meaning, and when to use it. Use plain text math notation.",
  "rules": [
    "Rule 1: key formula or theorem with the formula itself",
    "Rule 2: when and how to apply it",
    "Rule 3: common pitfall or edge case"
  ],
  "example_code": "A worked example problem with step-by-step solution (not code, but a solved problem)",
  "example_explanation": "Detailed walkthrough of each step in the example",
  "key_terms": ["term1", "term2", "term3"]
}}

Use plain text math notation: x^2 for powers, sqrt(x) for roots, dy/dx for derivatives, integral(f(x)dx) for integrals."""

    raw = llm.invoke(prompt)
    logger.info("[MATH] Lesson LLM response received")
    try:
        lesson = _parse_json_from_llm(raw)
        logger.info("[MATH] Lesson parsed — title: %s", lesson.get("title", "?")[:50])
    except (json.JSONDecodeError, ValueError):
        logger.warning("[MATH] Lesson JSON parse failed — using raw text")
        lesson = {
            "title": f"{topic}",
            "explanation": raw.strip(),
            "rules": [],
            "example_code": "",
            "example_explanation": "",
            "key_terms": [],
        }
    return lesson


def check_math_answer(session_id: str, user_answer: str) -> dict:
    """Check a math answer using LLM evaluation for equivalence.

    Math answers can be expressed in many equivalent forms (e.g. 1/2 = 0.5 = 50%),
    so we use LLM to judge equivalence rather than exact string matching.
    """
    logger.info("[MATH] Checking answer for session=%s", session_id)
    state = _tutor_sessions.get(session_id)
    if not state:
        logger.warning("[MATH] Session not found: %s", session_id)
        return {"error": "Session not found"}

    state["attempts"] += 1
    problem = state["problem"]
    style = state["style"]

    if style == "mcq":
        correct_letter = problem.get("correct_answer", "").strip().upper()
        user_letter = user_answer.strip().upper()
        if user_letter and correct_letter and user_letter[0] == correct_letter[0]:
            state["solved"] = True
            return {
                "correct": True,
                "feedback": problem.get("explanation", "Correct!"),
                "steps": problem.get("steps", []),
                "solved": True,
                "attempts": state["attempts"],
            }
        else:
            return {
                "correct": False,
                "feedback": "That's not right. Think about the steps carefully.",
                "solved": False,
                "attempts": state["attempts"],
            }

    # For solve/proof: use LLM to evaluate equivalence
    logger.info("[MATH] Using LLM to evaluate answer equivalence...")
    correct = problem.get("correct_answer", "")
    steps = problem.get("steps", [])
    llm = _llm(temperature=0.1)
    prompt = f"""You are grading a student's math answer.

Question: {problem.get('question', '')}
Correct answer: {correct}
Student's answer: {user_answer}

The student's answer may be in a different form (e.g., fraction vs decimal, different notation).
Evaluate if the student's answer is mathematically equivalent to the correct answer.

Return ONLY a JSON object:
{{
  "correct": true/false,
  "feedback": "Specific feedback on what they got right/wrong, referencing their work",
  "equivalence_note": "If almost correct, explain what's different"
}}"""

    raw = llm.invoke(prompt)
    try:
        result = _parse_json_from_llm(raw)
    except (json.JSONDecodeError, ValueError):
        result = {"correct": False, "feedback": raw.strip()}

    is_correct = result.get("correct", False)
    if is_correct:
        state["solved"] = True

    logger.info("[MATH] Answer check done — correct=%s, attempts=%d", is_correct, state["attempts"])
    return {
        "correct": is_correct,
        "feedback": result.get("feedback", ""),
        "steps": steps if is_correct else [],
        "solved": is_correct,
        "attempts": state["attempts"],
    }


def get_math_step_by_step(session_id: str) -> dict:
    """Return the full step-by-step solution for a math problem.

    Only available after solving or after max hints used.
    """
    logger.info("[MATH] Fetching step-by-step for session=%s", session_id)
    state = _tutor_sessions.get(session_id)
    if not state:
        logger.warning("[MATH] Session not found: %s", session_id)
        return {"error": "Session not found"}

    problem = state["problem"]
    steps = problem.get("steps", [])
    correct_answer = problem.get("correct_answer", "")

    if state["solved"] or state["attempts"] >= 3:
        return {
            "steps": steps,
            "correct_answer": correct_answer,
            "explanation": problem.get("explanation", ""),
        }

    return {
        "message": "Solve the problem or use hints first. Step-by-step available after 3 attempts.",
        "attempts": state["attempts"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MATH EVALUATION — interactive function evaluation for graphing
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_math_expression(expression: str, variable: str = "x", values: list = None) -> dict:
    """Safely evaluate a math expression at given values for interactive graphing.

    Supports: basic arithmetic, trig, log, exp, sqrt, powers.
    Returns {points: [{x, y}], derivative_points: [{x, y}], expression, derivative_expression}
    """
    logger.info("[MATH] Evaluating expression '%s' for graphing (%d points)", expression, len(values) if values else 41)
    import math as _math

    if values is None:
        values = [i * 0.5 for i in range(-20, 21)]  # -10 to 10 in 0.5 steps

    # Sanitize expression — only allow safe math operations
    _SAFE_NAMES = {
        'sin': _math.sin, 'cos': _math.cos, 'tan': _math.tan,
        'asin': _math.asin, 'acos': _math.acos, 'atan': _math.atan,
        'sinh': _math.sinh, 'cosh': _math.cosh, 'tanh': _math.tanh,
        'sqrt': _math.sqrt, 'abs': abs,
        'log': _math.log, 'log2': _math.log2, 'log10': _math.log10,
        'exp': _math.exp, 'pow': pow,
        'pi': _math.pi, 'e': _math.e,
        'floor': _math.floor, 'ceil': _math.ceil,
    }

    # Block anything with dunder, import, exec, eval, os, sys, etc.
    _BLOCKED = {'import', 'exec', 'eval', 'compile', 'open', 'os', 'sys',
                '__', 'globals', 'locals', 'getattr', 'setattr', 'delattr'}
    expr_lower = expression.lower()
    for blocked in _BLOCKED:
        if blocked in expr_lower:
            return {"error": f"Expression contains blocked keyword: {blocked}"}

    # Normalize expression for Python eval
    normalized = expression.replace('^', '**')

    # Try to compute a symbolic derivative using simple rules
    derivative_expr = _symbolic_derivative(expression, variable)

    points = []
    derivative_points = []

    for val in values:
        safe_globals = {"__builtins__": {}}
        safe_globals.update(_SAFE_NAMES)
        safe_globals[variable] = val

        try:
            y = eval(normalized, safe_globals)  # noqa: S307 — input is sanitized above
            if isinstance(y, (int, float)) and _math.isfinite(y):
                points.append({"x": val, "y": round(y, 6)})
        except Exception:
            pass

        if derivative_expr:
            try:
                deriv_normalized = derivative_expr.replace('^', '**')
                dy = eval(deriv_normalized, safe_globals)  # noqa: S307
                if isinstance(dy, (int, float)) and _math.isfinite(dy):
                    derivative_points.append({"x": val, "y": round(dy, 6)})
            except Exception:
                pass

    logger.info("[MATH] Expression evaluated — %d points, %d derivative points", len(points), len(derivative_points))
    return {
        "expression": expression,
        "derivative_expression": derivative_expr or "Could not compute symbolically",
        "points": points,
        "derivative_points": derivative_points,
        "variable": variable,
    }


def _symbolic_derivative(expression: str, variable: str = "x") -> str:
    """Attempt symbolic differentiation using pattern matching.

    Falls back to LLM for complex expressions.
    """
    expr = expression.strip()

    # Simple pattern-based derivatives
    if variable not in expr:
        return "0"
    if expr == variable:
        return "1"
    # x^n -> n*x^(n-1)
    m = re.match(rf'^{re.escape(variable)}\^(\d+)$', expr)
    if m:
        n = int(m.group(1))
        if n == 2:
            return f"2*{variable}"
        return f"{n}*{variable}^{n-1}"
    # a*x^n -> a*n*x^(n-1)
    m = re.match(rf'^(\d+)\*?{re.escape(variable)}\^(\d+)$', expr)
    if m:
        a, n = int(m.group(1)), int(m.group(2))
        coeff = a * n
        if n - 1 == 1:
            return f"{coeff}*{variable}"
        if n - 1 == 0:
            return str(coeff)
        return f"{coeff}*{variable}^{n-1}"
    # sin(x) -> cos(x), cos(x) -> -sin(x)
    if expr == f"sin({variable})":
        return f"cos({variable})"
    if expr == f"cos({variable})":
        return f"-sin({variable})"
    if expr == f"exp({variable})":
        return f"exp({variable})"
    if expr == f"log({variable})":
        return f"1/{variable}"

    # For complex expressions, use LLM
    try:
        llm = _llm(temperature=0.0)
        prompt = (
            f"Compute the derivative of f({variable}) = {expression} with respect to {variable}.\n"
            f"Return ONLY the derivative expression using plain text math notation "
            f"(^ for powers, sqrt() for roots, etc). No explanation."
        )
        result = llm.invoke(prompt).strip()
        result = re.sub(r"^f'?\(?\w\)?\s*=\s*", "", result)
        result = result.split('\n')[0].strip()
        if result and len(result) < 200:
            return result
    except Exception:
        pass

    return ""


def solve_matrix_problem(operation: str, matrices: list) -> dict:
    """Solve a matrix operation with step-by-step explanation.

    Supported operations: add, subtract, multiply, determinant, inverse, transpose
    """
    if not matrices:
        return {"error": "No matrices provided"}

    for i, m in enumerate(matrices):
        if not isinstance(m, list) or not all(isinstance(row, list) for row in m):
            return {"error": f"Matrix {i+1} is not a valid 2D array"}
        row_lens = [len(row) for row in m]
        if len(set(row_lens)) > 1:
            return {"error": f"Matrix {i+1} has inconsistent row lengths"}

    op = operation.lower().strip()
    steps = []

    try:
        if op in ("add", "subtract"):
            if len(matrices) < 2:
                return {"error": f"Need 2 matrices for {op}"}
            a, b = matrices[0], matrices[1]
            if len(a) != len(b) or len(a[0]) != len(b[0]):
                return {"error": "Matrices must have the same dimensions for addition/subtraction"}

            steps.append(f"Step 1: Verify dimensions match: {len(a)}x{len(a[0])} {'+'  if op == 'add' else '-'} {len(b)}x{len(b[0])}")
            sign = 1 if op == "add" else -1
            result = []
            for i in range(len(a)):
                row = []
                for j in range(len(a[0])):
                    val = a[i][j] + sign * b[i][j]
                    row.append(val)
                result.append(row)
            steps.append(f"Step 2: {'Add' if op == 'add' else 'Subtract'} corresponding elements")
            steps.append(f"Step 3: Result = {result}")
            return {"result": result, "steps": steps, "operation": op}

        elif op == "multiply":
            if len(matrices) < 2:
                return {"error": "Need 2 matrices for multiplication"}
            a, b = matrices[0], matrices[1]
            if len(a[0]) != len(b):
                return {"error": f"Cannot multiply: A is {len(a)}x{len(a[0])}, B is {len(b)}x{len(b[0])}. A's columns must equal B's rows."}

            steps.append(f"Step 1: Verify A({len(a)}x{len(a[0])}) * B({len(b)}x{len(b[0])}) -> Result will be {len(a)}x{len(b[0])}")
            result = [[0] * len(b[0]) for _ in range(len(a))]
            for i in range(len(a)):
                for j in range(len(b[0])):
                    total = sum(a[i][k] * b[k][j] for k in range(len(b)))
                    result[i][j] = total
            steps.append("Step 2: For each element (i,j): sum of row_i(A) * col_j(B)")
            steps.append(f"Step 3: Result = {result}")
            return {"result": result, "steps": steps, "operation": op}

        elif op == "determinant":
            m = matrices[0]
            if len(m) != len(m[0]):
                return {"error": "Determinant requires a square matrix"}

            def _det(mat: list) -> float:
                n = len(mat)
                if n == 1:
                    return mat[0][0]
                if n == 2:
                    return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
                total = 0
                for j in range(n):
                    minor = [row[:j] + row[j+1:] for row in mat[1:]]
                    total += ((-1) ** j) * mat[0][j] * _det(minor)
                return total

            det_val = _det(m)
            n = len(m)
            if n == 2:
                steps.append("Step 1: For 2x2 matrix [[a,b],[c,d]], det = ad - bc")
                steps.append(f"Step 2: det = {m[0][0]}*{m[1][1]} - {m[0][1]}*{m[1][0]}")
            else:
                steps.append("Step 1: Expand along the first row using cofactors")
                terms = []
                for j in range(n):
                    sign = "+" if j % 2 == 0 else "-"
                    terms.append(f"{sign} {m[0][j]} * M_{1}{j+1}")
                steps.append(f"Step 2: det = {' '.join(terms)}")
                steps.append("Step 3: Recursively compute each minor")
            steps.append(f"Result: det = {det_val}")
            return {"result": det_val, "steps": steps, "operation": op}

        elif op == "transpose":
            m = matrices[0]
            result = [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
            steps.append("Step 1: Swap rows and columns: element (i,j) becomes (j,i)")
            steps.append(f"Step 2: Original {len(m)}x{len(m[0])} -> Transposed {len(result)}x{len(result[0])}")
            steps.append(f"Result = {result}")
            return {"result": result, "steps": steps, "operation": op}

        elif op == "inverse":
            m = matrices[0]
            n = len(m)
            if n != len(m[0]):
                return {"error": "Inverse requires a square matrix"}

            def _det_inv(mat):
                sz = len(mat)
                if sz == 1:
                    return mat[0][0]
                if sz == 2:
                    return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
                total = 0
                for j in range(sz):
                    minor = [row[:j] + row[j+1:] for row in mat[1:]]
                    total += ((-1) ** j) * mat[0][j] * _det_inv(minor)
                return total

            det = _det_inv(m)
            if abs(det) < 1e-10:
                return {"error": "Matrix is singular (determinant = 0), no inverse exists",
                        "steps": ["Step 1: Compute determinant", f"det = {det}", "Matrix is not invertible."]}

            steps.append(f"Step 1: Compute determinant = {det}")

            if n == 2:
                inv = [
                    [m[1][1] / det, -m[0][1] / det],
                    [-m[1][0] / det, m[0][0] / det],
                ]
                steps.append("Step 2: For 2x2: swap diagonal, negate off-diagonal, divide by det")
                steps.append(f"Result = {[[round(v, 4) for v in row] for row in inv]}")
                return {"result": [[round(v, 6) for v in row] for row in inv], "steps": steps, "operation": op}

            # General case: adjugate method
            cofactors = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    minor = [row[:j] + row[j+1:] for row in (m[:i] + m[i+1:])]
                    cofactors[i][j] = ((-1) ** (i+j)) * _det_inv(minor)
            adj = [[cofactors[j][i] for j in range(n)] for i in range(n)]
            inv = [[adj[i][j] / det for j in range(n)] for i in range(n)]
            steps.append("Step 2: Compute cofactor matrix")
            steps.append("Step 3: Transpose to get adjugate matrix")
            steps.append("Step 4: Divide by determinant")
            steps.append(f"Result = {[[round(v, 4) for v in row] for row in inv]}")
            return {"result": [[round(v, 6) for v in row] for row in inv], "steps": steps, "operation": op}

        else:
            return {"error": f"Unsupported operation: {operation}. Supported: add, subtract, multiply, determinant, inverse, transpose"}

    except Exception as e:
        return {"error": str(e)}
