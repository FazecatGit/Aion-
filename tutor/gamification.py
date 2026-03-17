"""
Gamification Layer — XP, badges, streaks, and hint economy for tutor mode.

Persists to a JSON file so progress survives restarts.
"""

import json
import os
import logging
import time
import threading
from datetime import datetime, date
from typing import Optional
from pathlib import Path
from print_logger import get_logger

logger = get_logger("gamification")

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "cache", "gamification.json")
_state_lock = threading.RLock()

# ═══════════════════════════════════════════════════════════════════════════════
# XP REWARDS
# ═══════════════════════════════════════════════════════════════════════════════

XP_REWARDS = {
    "quiz_correct_first_try":  50,
    "quiz_correct":            25,
    "code_all_tests_pass":     75,
    "code_some_tests_pass":    15,
    "math_correct_first_try":  60,
    "math_correct":            30,
    "hint_used":              -10,   # spending XP for hints
    "streak_bonus_3":          30,
    "streak_bonus_7":         100,
    "streak_bonus_30":        500,
    "lesson_completed":        10,
    "chapter_completed":      100,
    "visualization_completed": 20,
}

# ═══════════════════════════════════════════════════════════════════════════════
# BADGE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

BADGES = {
    # Mastery badges — 3 correct in a row on a topic
    "topic_mastery":    {"name": "Concept Master",       "icon": "🏆", "desc": "3 correct answers in a row on one topic"},
    "math_mastery":     {"name": "Math Wizard",          "icon": "🧙", "desc": "3 math problems correct in a row"},
    "code_mastery":     {"name": "Code Ninja",           "icon": "🥷", "desc": "3 code problems with all tests passing in a row"},
    # Streak badges
    "streak_3":         {"name": "On Fire",              "icon": "🔥", "desc": "3-day learning streak"},
    "streak_7":         {"name": "Week Warrior",         "icon": "⚔️", "desc": "7-day learning streak"},
    "streak_30":        {"name": "Monthly Legend",       "icon": "👑", "desc": "30-day learning streak"},
    # Milestone badges
    "first_solve":      {"name": "First Steps",          "icon": "👣", "desc": "Solved your first problem"},
    "xp_100":           {"name": "Getting Started",      "icon": "⭐", "desc": "Earned 100 XP"},
    "xp_500":           {"name": "Rising Star",          "icon": "🌟", "desc": "Earned 500 XP"},
    "xp_1000":          {"name": "Knowledge Seeker",     "icon": "📚", "desc": "Earned 1000 XP"},
    "xp_5000":          {"name": "Scholar",              "icon": "🎓", "desc": "Earned 5000 XP"},
    "no_hints":         {"name": "No Training Wheels",   "icon": "🚀", "desc": "Solved 5 problems without using any hints"},
    "perfect_chapter":  {"name": "Perfect Chapter",      "icon": "💎", "desc": "Completed a chapter with no wrong answers"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _default_state() -> dict:
    return {
        "xp": 0,
        "level": 1,
        "badges": [],            # list of badge_id strings
        "streak_days": 0,
        "last_activity_date": None,
        "topic_streaks": {},     # topic -> consecutive correct count
        "no_hint_streak": 0,    # consecutive solves without hints
        "total_solved": 0,
        "total_attempted": 0,
        "chapter_scores": {},    # chapter_id -> {correct, wrong}
        "daily_xp": {},         # date_str -> xp earned that day
        "xp_history": [],       # [{date, amount, reason}] last 100
    }


_state: dict = {}


def _load_state():
    global _state
    with _state_lock:
        try:
            if os.path.isfile(_DATA_PATH):
                with open(_DATA_PATH, "r", encoding="utf-8") as f:
                    _state = json.load(f)
                logger.info("[GAMIFICATION] Loaded state: %d XP, level %d, %d badges",
                            _state.get("xp", 0), _state.get("level", 1), len(_state.get("badges", [])))
                return
        except Exception as e:
            logger.warning("[GAMIFICATION] Failed to load state: %s", e)
        _state = _default_state()


def _save_state():
    with _state_lock:
        try:
            os.makedirs(os.path.dirname(_DATA_PATH), exist_ok=True)
            with open(_DATA_PATH, "w", encoding="utf-8") as f:
                json.dump(_state, f, indent=2)
        except Exception as e:
            logger.warning("[GAMIFICATION] Failed to save state: %s", e)


# Initialize on import
_load_state()


def _level_for_xp(xp: int) -> int:
    """Compute level from total XP. Each level requires more XP than the last."""
    level = 1
    threshold = 100
    remaining = xp
    while remaining >= threshold:
        remaining -= threshold
        level += 1
        threshold = int(threshold * 1.3)
    return level


def _xp_for_next_level(xp: int) -> dict:
    """Return current level progress info."""
    level = 1
    threshold = 100
    remaining = xp
    while remaining >= threshold:
        remaining -= threshold
        level += 1
        threshold = int(threshold * 1.3)
    return {
        "level": level,
        "current_xp_in_level": remaining,
        "xp_needed_for_next": threshold,
        "progress_pct": round(remaining / threshold * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def award_xp(amount: int, reason: str, topic: str = "") -> dict:
    """Award XP and check for level-ups and badge unlocks."""
    if not _state:
        _load_state()

    old_level = _state.get("level", 1)
    _state["xp"] = max(0, _state.get("xp", 0) + amount)
    _state["level"] = _level_for_xp(_state["xp"])
    new_level = _state["level"]

    # Track daily XP
    today = date.today().isoformat()
    _state.setdefault("daily_xp", {})
    _state["daily_xp"][today] = _state["daily_xp"].get(today, 0) + max(0, amount)

    # Track history (last 100 entries)
    _state.setdefault("xp_history", [])
    _state["xp_history"].append({"date": today, "amount": amount, "reason": reason})
    if len(_state["xp_history"]) > 100:
        _state["xp_history"] = _state["xp_history"][-100:]

    leveled_up = new_level > old_level

    # Check XP milestone badges
    new_badges = []
    xp = _state["xp"]
    for threshold, badge_id in [(100, "xp_100"), (500, "xp_500"), (1000, "xp_1000"), (5000, "xp_5000")]:
        if xp >= threshold and badge_id not in _state.get("badges", []):
            new_badges.append(badge_id)

    for b in new_badges:
        _state.setdefault("badges", []).append(b)

    _save_state()

    result = {
        "xp_earned": amount,
        "total_xp": _state["xp"],
        "level": new_level,
        "leveled_up": leveled_up,
        "new_badges": [BADGES[b] | {"id": b} for b in new_badges],
    }
    if leveled_up:
        logger.info("[GAMIFICATION] LEVEL UP! %d -> %d (total XP: %d)", old_level, new_level, xp)
    return result


def record_solve(topic: str, correct: bool, used_hint: bool, is_math: bool = False, is_code: bool = False, first_try: bool = False, chapter_id: str = "") -> dict:
    """Record a problem solve attempt and compute XP + badges."""
    if not _state:
        _load_state()

    _state["total_attempted"] = _state.get("total_attempted", 0) + 1

    xp_amount = 0
    reason = ""
    new_badges = []

    if correct:
        _state["total_solved"] = _state.get("total_solved", 0) + 1

        # Compute XP
        if is_code:
            xp_amount = XP_REWARDS["code_all_tests_pass"] if first_try else XP_REWARDS["code_some_tests_pass"]
            reason = "Code problem solved"
        elif is_math:
            xp_amount = XP_REWARDS["math_correct_first_try"] if first_try else XP_REWARDS["math_correct"]
            reason = "Math problem solved"
        else:
            xp_amount = XP_REWARDS["quiz_correct_first_try"] if first_try else XP_REWARDS["quiz_correct"]
            reason = "Quiz answered correctly"

        # Topic mastery streak
        topic_key = topic.lower().strip()
        _state.setdefault("topic_streaks", {})
        _state["topic_streaks"][topic_key] = _state["topic_streaks"].get(topic_key, 0) + 1
        consec = _state["topic_streaks"][topic_key]
        if consec >= 3:
            badge_id = "math_mastery" if is_math else ("code_mastery" if is_code else "topic_mastery")
            if badge_id not in _state.get("badges", []):
                new_badges.append(badge_id)
                _state.setdefault("badges", []).append(badge_id)

        # No-hint streak
        if not used_hint:
            _state["no_hint_streak"] = _state.get("no_hint_streak", 0) + 1
            if _state["no_hint_streak"] >= 5 and "no_hints" not in _state.get("badges", []):
                new_badges.append("no_hints")
                _state.setdefault("badges", []).append("no_hints")
        else:
            _state["no_hint_streak"] = 0

        # First solve badge
        if _state["total_solved"] == 1 and "first_solve" not in _state.get("badges", []):
            new_badges.append("first_solve")
            _state.setdefault("badges", []).append("first_solve")

        # Chapter progress tracking
        if chapter_id:
            _state.setdefault("chapter_scores", {})
            _state["chapter_scores"].setdefault(chapter_id, {"correct": 0, "wrong": 0})
            _state["chapter_scores"][chapter_id]["correct"] += 1
    else:
        # Reset topic streak on wrong answer
        topic_key = topic.lower().strip()
        _state.setdefault("topic_streaks", {})
        _state["topic_streaks"][topic_key] = 0
        _state["no_hint_streak"] = 0

        if chapter_id:
            _state.setdefault("chapter_scores", {})
            _state["chapter_scores"].setdefault(chapter_id, {"correct": 0, "wrong": 0})
            _state["chapter_scores"][chapter_id]["wrong"] += 1

    # Update streak
    _update_streak()

    # Award XP
    xp_result = award_xp(xp_amount, reason, topic) if xp_amount else {
        "xp_earned": 0, "total_xp": _state.get("xp", 0),
        "level": _state.get("level", 1), "leveled_up": False, "new_badges": [],
    }

    # Merge new badges from both sources
    all_new = xp_result.get("new_badges", []) + [BADGES[b] | {"id": b} for b in new_badges]

    _save_state()

    return {
        "xp_earned": xp_result["xp_earned"],
        "total_xp": xp_result["total_xp"],
        "level": xp_result["level"],
        "leveled_up": xp_result["leveled_up"],
        "new_badges": all_new,
        "streak_days": _state.get("streak_days", 0),
        "topic_streak": _state.get("topic_streaks", {}).get(topic.lower().strip(), 0),
    }


def spend_xp_for_hint(cost: int = 10) -> dict:
    """Spend XP to reveal a hint. Returns success/failure and remaining XP."""
    if not _state:
        _load_state()

    current_xp = _state.get("xp", 0)
    if current_xp < cost:
        return {
            "success": False,
            "message": f"Not enough XP. Need {cost}, have {current_xp}.",
            "current_xp": current_xp,
        }

    _state["xp"] = current_xp - cost
    _state["level"] = _level_for_xp(_state["xp"])
    _save_state()

    return {
        "success": True,
        "xp_spent": cost,
        "current_xp": _state["xp"],
        "level": _state["level"],
    }


def _update_streak():
    """Update the daily learning streak."""
    today = date.today().isoformat()
    last = _state.get("last_activity_date")

    if last == today:
        return  # already active today

    if last:
        from datetime import timedelta
        last_date = date.fromisoformat(last)
        delta = (date.today() - last_date).days
        if delta == 1:
            _state["streak_days"] = _state.get("streak_days", 0) + 1
        elif delta > 1:
            _state["streak_days"] = 1  # streak broken, restart
    else:
        _state["streak_days"] = 1

    _state["last_activity_date"] = today

    # Check streak badges
    streak = _state["streak_days"]
    new_badges = []
    for threshold, badge_id, bonus_key in [(3, "streak_3", "streak_bonus_3"),
                                            (7, "streak_7", "streak_bonus_7"),
                                            (30, "streak_30", "streak_bonus_30")]:
        if streak >= threshold and badge_id not in _state.get("badges", []):
            _state.setdefault("badges", []).append(badge_id)
            new_badges.append(badge_id)
            award_xp(XP_REWARDS[bonus_key], f"{threshold}-day streak bonus!")
            logger.info("[GAMIFICATION] Streak badge unlocked: %s (%d days)", badge_id, streak)


def get_profile() -> dict:
    """Return the full gamification profile."""
    if not _state:
        _load_state()

    level_info = _xp_for_next_level(_state.get("xp", 0))
    earned_badges = [
        BADGES[b] | {"id": b}
        for b in _state.get("badges", [])
        if b in BADGES
    ]

    return {
        "xp": _state.get("xp", 0),
        "level": level_info["level"],
        "level_progress": level_info,
        "badges": earned_badges,
        "all_badges": [BADGES[b] | {"id": b} for b in BADGES],
        "streak_days": _state.get("streak_days", 0),
        "last_activity_date": _state.get("last_activity_date"),
        "total_solved": _state.get("total_solved", 0),
        "total_attempted": _state.get("total_attempted", 0),
        "topic_streaks": _state.get("topic_streaks", {}),
        "daily_xp": _state.get("daily_xp", {}),
    }


def reset_profile() -> dict:
    """Reset all gamification progress."""
    global _state
    _state = _default_state()
    _save_state()
    return {"status": "ok", "message": "Gamification progress reset"}
