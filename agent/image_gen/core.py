"""Shared globals, progress tracking, VRAM management, and cancellation."""

import json
import logging
import threading
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

from brain.config import OUTPUT_DIR

_log = logging.getLogger("image_gen")

# ── Model management ──────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
SUPPORTED_EXTENSIONS = {".safetensors", ".ckpt"}
LORA_EXTENSIONS = {".safetensors"}

# Cached pipelines and active model state
_loaded_pipelines: dict[str, StableDiffusionXLPipeline] = {}
_active_model: str | None = None
_active_loras: list[str] = []

# ── Global cancel flag ────────────────────────────────────────────────────
_cancel_event = threading.Event()

# ── Generation progress tracking ─────────────────────────────────────────
_generation_progress: dict = {
    "active": False,
    "type": "idle",
    "current_step": 0,
    "total_steps": 0,
    "current_frame": 0,
    "total_frames": 0,
    "vram_used_mb": 0,
    "vram_total_mb": 0,
    "vram_percent": 0.0,
    "message": "",
}
_progress_lock = threading.Lock()

# ── Video generation directories ─────────────────────────────────────────
VIDEO_OUTPUT_DIR = Path(OUTPUT_DIR) / "videos"
VIDEO_CHECKPOINT_DIR = Path("cache") / "video_checkpoints"
VIDEO_QUEUE_FILE = Path("cache") / "video_queue.json"

# ── Video pause/resume event ─────────────────────────────────────────────
_video_pause_event = threading.Event()
_video_generating = threading.Event()

# VRAM safety threshold
_VRAM_SAFETY_PERCENT = 92.0


# ── Utility functions ────────────────────────────────────────────────────

def _save_json(path: Path, data: dict):
    """Atomic-ish JSON write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(path)


def _update_progress(**kwargs):
    """Thread-safe update of generation progress dict."""
    with _progress_lock:
        _generation_progress.update(kwargs)


def get_generation_progress() -> dict:
    """Return a snapshot of current generation progress + GPU stats."""
    with _progress_lock:
        snap = dict(_generation_progress)
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            snap["vram_used_mb"] = round(allocated, 1)
            snap["vram_total_mb"] = round(total, 1)
            snap["vram_percent"] = round(allocated / total * 100, 1) if total > 0 else 0
        except Exception:
            pass
    return snap


def get_gpu_info() -> dict:
    """Return GPU VRAM stats."""
    if not torch.cuda.is_available():
        return {"available": False, "vram_used_mb": 0, "vram_total_mb": 0,
                "vram_percent": 0, "device_name": "N/A"}
    try:
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        name = torch.cuda.get_device_name(0)
        return {
            "available": True,
            "device_name": name,
            "vram_used_mb": round(allocated, 1),
            "vram_reserved_mb": round(reserved, 1),
            "vram_total_mb": round(total, 1),
            "vram_percent": round(allocated / total * 100, 1) if total > 0 else 0,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def _check_vram_safe() -> bool:
    """Return False if VRAM usage exceeds safety threshold."""
    if not torch.cuda.is_available():
        return True
    try:
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        pct = allocated / total * 100 if total > 0 else 0
        return pct < _VRAM_SAFETY_PERCENT
    except Exception:
        return True


def cancel_generation():
    """Signal all running generation loops to stop after the current step."""
    _cancel_event.set()
    _video_pause_event.clear()
    return {"status": "ok", "message": "Cancellation requested"}


def _check_cancelled():
    """Check if cancellation was requested and clear the flag."""
    if _cancel_event.is_set():
        _cancel_event.clear()
        return True
    return False


def _recover_interrupted_jobs():
    """On server start, mark any 'running' jobs as 'interrupted'."""
    if not VIDEO_CHECKPOINT_DIR.exists():
        return
    for job_dir in VIDEO_CHECKPOINT_DIR.iterdir():
        meta_file = job_dir / "job.json"
        if not meta_file.exists():
            continue
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("status") in ("running", "paused"):
                meta["status"] = "interrupted"
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, default=str)
                _log.info("[WAN VIDEO] Recovered interrupted job: %s (step %d/%d)",
                          meta.get("job_id"), meta.get("last_completed_step", 0),
                          meta.get("steps", 0))
        except Exception:
            pass

try:
    _recover_interrupted_jobs()
except Exception:
    pass

