"""
Agent metrics collection — tracks LLM calls, timing, token usage,
success rates, and strategy effectiveness across agent operations.

Usage:
    with MetricsCollector("edit_code") as mc:
        # ... do work ...
        mc.record_llm_call(model="qwen3", prompt_tokens=500, output_tokens=200, elapsed=3.2)
        mc.record_event("self_repair_triggered")
    summary = mc.summary()
"""

import time
import threading
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json

logger = logging.getLogger("agent_metrics")

# ── Context-var so TimedLLM can push to the active collector ────────────────
_active_collector: ContextVar[Optional["MetricsCollector"]] = ContextVar(
    "_active_collector", default=None
)


def get_active_collector() -> Optional["MetricsCollector"]:
    return _active_collector.get()


@dataclass
class LLMCallRecord:
    call_id: int
    model: str
    prompt_tokens: int
    output_tokens: int
    elapsed: float
    success: bool
    label: str = ""


@dataclass
class MetricsCollector:
    """Collects metrics for a single agent operation (edit, test-fix, multi-agent run)."""

    operation: str
    llm_calls: list = field(default_factory=list)
    events: list = field(default_factory=list)
    _start: float = 0.0
    _end: float = 0.0
    _token: object = field(default=None, repr=False)

    def __enter__(self):
        self._start = time.monotonic()
        self._token = _active_collector.set(self)
        return self

    def __exit__(self, *exc):
        self._end = time.monotonic()
        _active_collector.reset(self._token)

    def record_llm_call(self, call_id: int, model: str, prompt_tokens: int,
                        output_tokens: int, elapsed: float, success: bool = True,
                        label: str = ""):
        self.llm_calls.append(LLMCallRecord(
            call_id=call_id, model=model, prompt_tokens=prompt_tokens,
            output_tokens=output_tokens, elapsed=elapsed, success=success,
            label=label,
        ))

    def record_event(self, name: str, **data):
        self.events.append({"name": name, "time": time.monotonic() - self._start, **data})

    @property
    def wall_time(self) -> float:
        end = self._end or time.monotonic()
        return end - self._start

    def summary(self) -> dict:
        total_prompt = sum(c.prompt_tokens for c in self.llm_calls)
        total_output = sum(c.output_tokens for c in self.llm_calls)
        total_llm_time = sum(c.elapsed for c in self.llm_calls)
        failed = [c for c in self.llm_calls if not c.success]

        return {
            "operation": self.operation,
            "wall_time_s": round(self.wall_time, 2),
            "llm_calls": len(self.llm_calls),
            "llm_calls_failed": len(failed),
            "total_prompt_tokens": total_prompt,
            "total_output_tokens": total_output,
            "total_tokens": total_prompt + total_output,
            "total_llm_time_s": round(total_llm_time, 2),
            "avg_llm_latency_s": round(total_llm_time / len(self.llm_calls), 2) if self.llm_calls else 0,
            "events": self.events,
        }


# ── Persistent metrics store (append-only JSONL) ───────────────────────────
_METRICS_FILE = Path(__file__).resolve().parent.parent / "cache" / "agent_metrics.jsonl"
_file_lock = threading.Lock()


def persist_summary(summary: dict):
    """Append a metrics summary to the JSONL log file."""
    summary["timestamp"] = time.time()
    try:
        with _file_lock:
            _METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(_METRICS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary) + "\n")
    except OSError as e:
        logger.warning("Failed to persist metrics: %s", e)


def load_history(last_n: int = 50) -> list[dict]:
    """Load the most recent N metric summaries."""
    if not _METRICS_FILE.exists():
        return []
    lines = _METRICS_FILE.read_text(encoding="utf-8").strip().splitlines()
    results = []
    for line in lines[-last_n:]:
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return results


def aggregate_stats(history: list[dict] | None = None) -> dict:
    """Compute aggregate statistics from persisted history."""
    if history is None:
        history = load_history(last_n=500)
    if not history:
        return {"total_operations": 0}

    total_ops = len(history)
    total_llm = sum(h.get("llm_calls", 0) for h in history)
    total_tokens = sum(h.get("total_tokens", 0) for h in history)
    total_wall = sum(h.get("wall_time_s", 0) for h in history)
    total_failed = sum(h.get("llm_calls_failed", 0) for h in history)

    ops_by_type: dict[str, int] = {}
    for h in history:
        op = h.get("operation", "unknown")
        ops_by_type[op] = ops_by_type.get(op, 0) + 1

    return {
        "total_operations": total_ops,
        "total_llm_calls": total_llm,
        "total_tokens": total_tokens,
        "total_wall_time_s": round(total_wall, 1),
        "avg_wall_time_s": round(total_wall / total_ops, 2),
        "avg_llm_calls_per_op": round(total_llm / total_ops, 1),
        "avg_tokens_per_op": round(total_tokens / total_ops, 0),
        "failed_llm_calls": total_failed,
        "operations_by_type": ops_by_type,
    }
