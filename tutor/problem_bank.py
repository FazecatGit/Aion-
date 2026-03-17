"""
Problem Bank — verified problems stored in RAG instead of LLM-generated from scratch.

Pattern: model as tutor, documents as curriculum.
The model retrieves real problems with verified answers, explains them,
and can vary the surface (change numbers/context). Verification is trivial
because the answer already exists in the document.
"""

import json
import os
import re
import random
import logging
from typing import Optional
from print_logger import get_logger

logger = get_logger("problem_bank")

_BANK_DIR = os.path.join(os.path.dirname(__file__), "..", "cache", "problem_bank")

# ═══════════════════════════════════════════════════════════════════════════════
# PROBLEM BANK STORAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_bank_dir():
    os.makedirs(_BANK_DIR, exist_ok=True)


def _bank_file(category: str) -> str:
    safe = re.sub(r'[^\w\-]', '_', category.lower().strip())
    return os.path.join(_BANK_DIR, f"{safe}.json")


def _load_bank(category: str) -> list[dict]:
    path = _bank_file(category)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_bank(category: str, problems: list[dict]):
    _ensure_bank_dir()
    with open(_bank_file(category), "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# CRUD
# ═══════════════════════════════════════════════════════════════════════════════

def add_problem(
    category: str,
    topic: str,
    difficulty: str,
    question: str,
    correct_answer: str,
    style: str = "solve",
    *,
    options: list[str] | None = None,
    steps: list[str] | None = None,
    hints: list[str] | None = None,
    test_cases: list[dict] | None = None,
    source: str = "manual",
    tags: list[str] | None = None,
) -> dict:
    """Add a verified problem to the bank."""
    problems = _load_bank(category)

    problem = {
        "id": f"{category}_{len(problems)}_{random.randint(1000, 9999)}",
        "topic": topic,
        "difficulty": difficulty,
        "style": style,
        "question": question,
        "correct_answer": correct_answer,
        "options": options or [],
        "steps": steps or [],
        "hints": hints or [],
        "test_cases": test_cases or [],
        "source": source,
        "tags": tags or [],
        "times_served": 0,
        "times_correct": 0,
    }

    problems.append(problem)
    _save_bank(category, problems)
    logger.info("[BANK] Added problem '%s' to category '%s'", problem["id"], category)
    return problem


def get_problem(
    category: str = "",
    topic: str = "",
    difficulty: str = "",
    style: str = "",
    exclude_ids: list[str] | None = None,
) -> Optional[dict]:
    """Retrieve a problem from the bank, optionally filtered.

    Prioritizes least-served problems to ensure variety.
    """
    candidates = []

    # Search across categories if none specified
    categories = [category] if category else _list_categories()
    for cat in categories:
        problems = _load_bank(cat)
        for p in problems:
            if topic and topic.lower() not in p.get("topic", "").lower():
                continue
            if difficulty and p.get("difficulty", "").lower() != difficulty.lower():
                continue
            if style and p.get("style", "").lower() != style.lower():
                continue
            if exclude_ids and p.get("id") in exclude_ids:
                continue
            candidates.append((cat, p))

    if not candidates:
        return None

    # Sort by times_served (ascending) to spread problem distribution
    candidates.sort(key=lambda x: x[1].get("times_served", 0))
    # Pick from the least-served tier with some randomness
    min_served = candidates[0][1].get("times_served", 0)
    least_served = [(c, p) for c, p in candidates if p.get("times_served", 0) <= min_served + 1]
    cat, chosen = random.choice(least_served)

    # Increment serve count
    chosen["times_served"] = chosen.get("times_served", 0) + 1
    problems = _load_bank(cat)
    for p in problems:
        if p.get("id") == chosen.get("id"):
            p["times_served"] = chosen["times_served"]
            break
    _save_bank(cat, problems)

    logger.info("[BANK] Serving problem '%s' from '%s' (served %d times)",
                chosen.get("id"), cat, chosen["times_served"])
    return chosen


def record_attempt(problem_id: str, correct: bool):
    """Record whether a banked problem was answered correctly."""
    for cat in _list_categories():
        problems = _load_bank(cat)
        for p in problems:
            if p.get("id") == problem_id:
                if correct:
                    p["times_correct"] = p.get("times_correct", 0) + 1
                _save_bank(cat, problems)
                return


def get_bank_stats() -> dict:
    """Return statistics about the problem bank."""
    total = 0
    by_category = {}
    by_difficulty = {}
    for cat in _list_categories():
        problems = _load_bank(cat)
        total += len(problems)
        by_category[cat] = len(problems)
        for p in problems:
            d = p.get("difficulty", "unknown")
            by_difficulty[d] = by_difficulty.get(d, 0) + 1

    return {
        "total_problems": total,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "categories": _list_categories(),
    }


def _list_categories() -> list[str]:
    _ensure_bank_dir()
    cats = []
    for f in os.listdir(_BANK_DIR):
        if f.endswith(".json"):
            cats.append(f[:-5])
    return cats


# ═══════════════════════════════════════════════════════════════════════════════
# RAG INTEGRATION — extract problems from ingested documents
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_problems_from_rag(topic: str, max_problems: int = 5) -> list[dict]:
    """Search RAG for real problems/exercises and add them to the bank.

    Uses the model to EXPLAIN retrieved problems, not to GENERATE new ones.
    The verified answer comes from the document, not from the model's generation.
    """
    try:
        from brain.fast_search import fast_topic_search
        from brain.config import make_llm
    except ImportError:
        logger.warning("[BANK] Cannot import RAG modules")
        return []

    results = fast_topic_search(
        f"exercise problem example {topic}",
        top_k=10,
    )
    if not results:
        logger.info("[BANK] No RAG results for topic '%s'", topic)
        return []

    llm = make_llm(temperature=0.1)
    added = []

    for doc in results[:max_problems]:
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        if len(content.strip()) < 50:
            continue

        # Ask LLM to extract a problem from the document chunk — not generate one
        prompt = (
            f"The following text is from a textbook or reference document. "
            f"Extract ONE practice problem with its answer from this text. "
            f"If the text contains a worked example, convert it into a practice problem.\n\n"
            f"TEXT:\n{content[:1500]}\n\n"
            f"Return ONLY a JSON object:\n"
            f'{{\n'
            f'  "question": "The extracted/adapted problem statement",\n'
            f'  "correct_answer": "The verified answer from the text",\n'
            f'  "steps": ["Step 1: ...", "Step 2: ..."],\n'
            f'  "difficulty": "easy|medium|hard",\n'
            f'  "style": "solve|mcq|proof",\n'
            f'  "hints": ["Hint 1", "Hint 2"]\n'
            f'}}\n\n'
            f"If no problem can be extracted, return {{}}"
        )

        try:
            raw = llm.invoke(prompt)
            # Simple JSON extraction
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())
            brace_start = raw.find("{")
            brace_end = raw.rfind("}")
            if brace_start != -1 and brace_end > brace_start:
                parsed = json.loads(raw[brace_start:brace_end + 1])
                if parsed.get("question") and parsed.get("correct_answer"):
                    problem = add_problem(
                        category=topic.lower().replace(" ", "_"),
                        topic=topic,
                        difficulty=parsed.get("difficulty", "medium"),
                        question=parsed["question"],
                        correct_answer=parsed["correct_answer"],
                        style=parsed.get("style", "solve"),
                        steps=parsed.get("steps"),
                        hints=parsed.get("hints"),
                        source="rag_extracted",
                        tags=[topic],
                    )
                    added.append(problem)
        except Exception as e:
            logger.debug("[BANK] Failed to extract problem from chunk: %s", e)
            continue

    logger.info("[BANK] Ingested %d problems from RAG for topic '%s'", len(added), topic)
    return added
