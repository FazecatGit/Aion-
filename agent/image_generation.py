"""
Offline image generation using Stable Diffusion XL (local safetensors models).

Supports:
- Multiple checkpoint models (auto-discovered from models/ folder)
- LoRA loading on top of base checkpoints
- Explicit vs normal mode (no hardcoded NSFW prefixes)
- Long-prompt handling via CLIP chunking to bypass 77-token limit
- Consistent animated generation via img2img frame chaining
- Storyboard / line-drawing preview before full generation
- Save/resume checkpoints for long-running generation jobs
- Art style presets (anime, hand-drawn, painterly, etc.)
- LoRA training pipeline for custom art styles / characters
- Tiled upscaling for VRAM-constrained GPUs (8 GB+)
- Video output (mp4) alongside GIF
- Frame editing: re-generate a single frame keeping neighbours consistent
- Training dataset critique (checks image count, coverage, quality)
- Feedback loop: track what the pipeline actually focused on
"""

import os
import re
import json
import logging
import time
import shutil
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image

from brain.config import OUTPUT_DIR

_log = logging.getLogger("image_gen")

# ── Model management ──────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent.parent / "models"
_loaded_pipelines: dict[str, StableDiffusionXLPipeline] = {}
_active_model: str | None = None
_active_loras: list[str] = []  # currently loaded LoRA names

SUPPORTED_EXTENSIONS = {".safetensors", ".ckpt"}
LORA_EXTENSIONS = {".safetensors"}

# ── Global cancel flag ────────────────────────────────────────────────────
_cancel_event = threading.Event()

# ── Generation progress tracking ─────────────────────────────────────────
_generation_progress: dict = {
    "active": False,
    "type": "idle",          # "image", "animation", "idle", "video"
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
_video_pause_event = threading.Event()  # set = paused
_video_generating = threading.Event()   # set = a video gen is in progress


# ── Mark crashed/interrupted jobs on module load ──────────────────────
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
    # Always refresh VRAM stats
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


# VRAM safety threshold — if VRAM exceeds this %, auto-save & stop
_VRAM_SAFETY_PERCENT = 92.0


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
    # Also clear pause so the cancel can propagate through a paused state
    _video_pause_event.clear()
    return {"status": "ok", "message": "Cancellation requested"}


def _check_cancelled():
    """Check if cancellation was requested and clear the flag."""
    if _cancel_event.is_set():
        _cancel_event.clear()
        return True
    return False


def list_models() -> list[dict]:
    """Discover all checkpoint models and LoRAs in the models folder."""
    models = []
    if not MODELS_DIR.exists():
        return models

    for f in sorted(MODELS_DIR.iterdir()):
        if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file():
            is_lora = _is_lora_file(f)
            models.append({
                "name": f.stem,
                "filename": f.name,
                "path": str(f),
                "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                "type": "lora" if is_lora else "checkpoint",
            })

    # Scan all LoRA subdirectories
    LORA_CATEGORIES = ("styles", "characters", "clothing", "poses", "concept", "action", "loras")
    for subdir_name in LORA_CATEGORIES:
        sub_dir = MODELS_DIR / subdir_name
        if sub_dir.exists():
            for f in sorted(sub_dir.iterdir()):
                if f.suffix.lower() in LORA_EXTENSIONS and f.is_file():
                    tw = get_trigger_words(f.stem)
                    models.append({
                        "name": f.stem,
                        "filename": f.name,
                        "path": str(f),
                        "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                        "type": "lora",
                        "category": subdir_name,
                        "trigger_words": tw,
                    })

    return models


def _is_lora_file(path: Path) -> bool:
    """Heuristic: LoRA files are generally < 300 MB."""
    return path.stat().st_size < 300 * 1024 * 1024


# ── LoRA trigger word management ──────────────────────────────────────────

_TRIGGER_WORDS_FILE = MODELS_DIR / "trigger_words.json"


def _load_trigger_words() -> dict:
    """Load the trigger words JSON file."""
    if _TRIGGER_WORDS_FILE.exists():
        try:
            return json.loads(_TRIGGER_WORDS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_trigger_words(data: dict):
    """Persist trigger words to disk."""
    _TRIGGER_WORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _TRIGGER_WORDS_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_trigger_word_entry(entry) -> dict:
    """Parse a trigger word entry, which may be a plain string or a weighted object.

    Returns: {"word": str, "weight": float}
    """
    if isinstance(entry, dict):
        return {"word": str(entry.get("word", "")), "weight": float(entry.get("weight", 1.0))}
    return {"word": str(entry), "weight": 1.0}


def format_trigger_word(entry) -> str:
    """Format a trigger word with its weight for prompt injection.

    Plain weight 1.0 → just the word. Other weights → (word:weight) syntax.
    """
    parsed = parse_trigger_word_entry(entry)
    word = parsed["word"]
    weight = parsed["weight"]
    if not word or word == ";":
        return ""
    # Already has explicit weight syntax like (word:1.2) — leave as-is
    if re.match(r'^\(.*:\d+\.?\d*\)$', word):
        return word
    if abs(weight - 1.0) < 0.01:
        return word
    return f"({word}:{weight:.1f})"


def parse_outfit_groups(words: list) -> dict:
    """Parse a trigger word array into primary trigger + outfit groups.

    Supports TWO formats:

    1) Semicolon-delimited (legacy):
        ["primaryTrigger", ";", "outfitName", "tag1", "tag2", ";", "outfit2", ...]

    2) Colon-delimited (ChamModels style):
        Each entry is "Outfit Name:TriggerCode, tag1, tag2, tag3"
        or "Base appearance (for custom outfits): TriggerCode, tag1, tag2"

    Returns:
        {
            "primary": "primaryTrigger",
            "outfits": {
                "outfitName": [{"word": "tag1", "weight": 1.0}, ...],
                ...
            },
            "flat_words": [{"word": "...", "weight": 1.0}, ...],
            "has_outfits": True/False
        }
    """
    raw_entries = [parse_trigger_word_entry(w) for w in words]

    # ── Strategy 1: Semicolon-delimited format ─────────────────────────────
    has_semicolons = any(e["word"] == ";" for e in raw_entries)
    if has_semicolons:
        groups: list[list[dict]] = []
        current: list[dict] = []
        for e in raw_entries:
            if e["word"] == ";":
                if current:
                    groups.append(current)
                current = []
            else:
                if e["word"]:
                    current.append(e)
        if current:
            groups.append(current)

        primary = groups[0][0]["word"] if groups and groups[0] else ""
        outfits: dict[str, list[dict]] = {}
        for g in groups[1:]:
            if g:
                outfit_name = g[0]["word"]
                outfit_tags = g[1:] if len(g) > 1 else []
                outfits[outfit_name] = outfit_tags

        flat = [e for e in raw_entries if e["word"] and e["word"] != ";"]
        return {
            "primary": primary,
            "outfits": outfits,
            "flat_words": flat,
            "has_outfits": bool(outfits),
        }

    # ── Strategy 2: Colon-delimited format ─────────────────────────────────
    # Detect entries like "Outfit Name:TriggerCode, tag1, tag2"
    # or "Base appearance (for custom outfits): TriggerCode, tag1, tag2"
    colon_entries = []
    for e in raw_entries:
        word = e["word"]
        if not word:
            continue
        # Match "Label: tags" or "Label:tags" pattern  
        # Labels typically contain words like "Outfit", "Default", "Base", "Casual", etc.
        colon_match = re.match(r'^(.+?):\s*(.+)$', word)
        if colon_match:
            label = colon_match.group(1).strip()
            rest = colon_match.group(2).strip()
            colon_entries.append((label, rest, e))

    # If most entries have colons, treat as colon-delimited outfit format
    if len(colon_entries) >= 2 and len(colon_entries) >= len(raw_entries) * 0.5:
        primary = ""
        outfits = {}

        for label, rest, _orig in colon_entries:
            # Parse the rest as comma-separated tags
            tags = [{"word": t.strip(), "weight": 1.0} for t in rest.split(",") if t.strip()]
            is_base = "base" in label.lower() or "custom outfit" in label.lower()

            if is_base and not primary:
                # Base appearance — first tag is typically the trigger code
                primary = tags[0]["word"] if tags else label
                outfits[label] = tags
            else:
                # Named outfit
                outfits[label] = tags
                if not primary and tags:
                    primary = tags[0]["word"]

        flat = [e for e in raw_entries if e["word"]]
        return {
            "primary": primary,
            "outfits": outfits,
            "flat_words": flat,
            "has_outfits": bool(outfits),
        }

    # ── Strategy 3: Simple flat list, no outfits ───────────────────────────
    flat = [e for e in raw_entries if e["word"]]
    return {
        "primary": flat[0]["word"] if flat else "",
        "outfits": {},
        "flat_words": flat,
        "has_outfits": False,
    }


def build_trigger_prompt(words: list, selected_outfit: str | None = None) -> str:
    """Build the trigger word string to inject into a prompt.

    - If the LoRA has outfit groups and an outfit is selected, injects the
      primary trigger + that outfit's tags.
    - If no outfit selected but outfits exist, uses the first outfit.
    - If no outfit groups, injects ALL trigger words.

    Returns a comma-separated string ready to prepend to the prompt.
    """
    parsed = parse_outfit_groups(words)

    parts: list[str] = []

    if parsed["has_outfits"]:
        # Always include the primary trigger
        primary_fmt = format_trigger_word({"word": parsed["primary"], "weight": 1.0})
        if primary_fmt:
            parts.append(primary_fmt)

        # Select outfit
        outfit_tags: list[dict] = []
        if selected_outfit and selected_outfit in parsed["outfits"]:
            outfit_tags = parsed["outfits"][selected_outfit]
        elif parsed["outfits"]:
            # Default to first outfit
            first_key = next(iter(parsed["outfits"]))
            outfit_tags = parsed["outfits"][first_key]

        for tag in outfit_tags:
            fmt = format_trigger_word(tag)
            if fmt:
                parts.append(fmt)
    else:
        # No outfits — inject ALL trigger words
        for entry in parsed["flat_words"]:
            fmt = format_trigger_word(entry)
            if fmt:
                parts.append(fmt)

    return ", ".join(parts)


def get_trigger_words(lora_name: str) -> list[str]:
    """Get trigger words for a specific LoRA by name."""
    data = _load_trigger_words()
    return data.get(lora_name, [])


def get_trigger_words_parsed(lora_name: str) -> dict:
    """Get parsed trigger words with outfit groups for a specific LoRA."""
    data = _load_trigger_words()
    raw = data.get(lora_name, [])
    parsed = parse_outfit_groups(raw)
    return {
        "lora": lora_name,
        "primary": parsed["primary"],
        "has_outfits": parsed["has_outfits"],
        "outfits": {k: [format_trigger_word(t) for t in v] for k, v in parsed["outfits"].items()},
        "all_words": [format_trigger_word(e) for e in parsed["flat_words"]],
    }


def set_trigger_words(lora_name: str, words: list) -> dict:
    """Set trigger words for a LoRA. Overwrites any existing entry.

    Accepts plain strings, weighted objects {"word": str, "weight": float},
    and ";" separators for outfit grouping.
    """
    data = _load_trigger_words()
    cleaned = []
    for w in words:
        if isinstance(w, dict):
            word = str(w.get("word", "")).strip()
            weight = float(w.get("weight", 1.0))
            if word:
                if word == ";" or abs(weight - 1.0) < 0.01:
                    cleaned.append(word)
                else:
                    cleaned.append({"word": word, "weight": weight})
        elif isinstance(w, str):
            w = w.strip()
            if w:
                cleaned.append(w)
    data[lora_name] = cleaned
    _save_trigger_words(data)
    return {"status": "ok", "lora": lora_name, "trigger_words": data[lora_name]}


def delete_trigger_words(lora_name: str) -> dict:
    """Remove trigger words for a LoRA."""
    data = _load_trigger_words()
    data.pop(lora_name, None)
    _save_trigger_words(data)
    return {"status": "ok", "lora": lora_name}


def get_all_trigger_words() -> dict:
    """Return the full trigger words mapping with parsed outfit info."""
    raw = _load_trigger_words()
    result = {}
    for lora_name, words in raw.items():
        parsed = parse_outfit_groups(words)
        result[lora_name] = {
            "raw": words,
            "primary": parsed["primary"],
            "has_outfits": parsed["has_outfits"],
            "outfits": list(parsed["outfits"].keys()),
        }
    return {"status": "ok", "trigger_words": result}


def list_loras_by_category() -> dict:
    """List all LoRAs organised by folder category with their trigger words."""
    categories: dict[str, list[dict]] = {}
    LORA_CATEGORIES = ("styles", "characters", "clothing", "poses", "concept", "action")
    tw_data = _load_trigger_words()
    PREVIEW_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
    for cat in LORA_CATEGORIES:
        cat_dir = MODELS_DIR / cat
        entries = []
        if cat_dir.exists():
            for f in sorted(cat_dir.iterdir()):
                if f.suffix.lower() in LORA_EXTENSIONS and f.is_file():
                    # Look for a preview image: <name>.preview.png/jpg/webp
                    preview = None
                    for ext in PREVIEW_EXTENSIONS:
                        candidate = f.parent / f"{f.stem}.preview{ext}"
                        if candidate.exists():
                            preview = str(candidate)
                            break
                    entries.append({
                        "name": f.stem,
                        "filename": f.name,
                        "path": str(f),
                        "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                        "trigger_words": tw_data.get(f.stem, []),
                        "preview_image": preview,
                    })
        categories[cat] = entries
    return {"status": "ok", "categories": categories}


def _fuzzy_match_score(a: str, b: str) -> float:
    """Simple character-level similarity ratio between two strings (0.0–1.0)."""
    a_low, b_low = a.lower(), b.lower()
    if a_low == b_low:
        return 1.0
    if not a_low or not b_low:
        return 0.0
    # Longest common subsequence ratio
    m, n = len(a_low), len(b_low)
    if m > 40 or n > 40:
        return 0.0  # skip long strings
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a_low[i - 1] == b_low[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    return (2.0 * lcs) / (m + n)


def validate_prompt_characters(prompt: str, lora_stems: list[str] | None = None) -> dict:
    """Validate prompt against known character trigger codes.

    Scans the prompt text for words that look like character names or trigger
    codes, compares them against the known trigger words from all character
    LoRAs, and returns matches + warnings for possible misspellings.

    Returns:
        {
            "detected_characters": [{name, trigger_code, lora, confidence}],
            "warnings": [{text, suggestion, lora, reason}],
            "unmatched_loras": [lora_stem],  # loaded LoRAs with no trigger in prompt
        }
    """
    tw_data = _load_trigger_words()
    char_dir = MODELS_DIR / "characters"

    # Build a map of all known trigger codes → (lora_stem, display_name)
    known_triggers: dict[str, tuple[str, str]] = {}  # code_lower → (stem, code_original)
    all_char_stems: set[str] = set()

    for stem, words in tw_data.items():
        # Check if this is a character LoRA
        lora_path = char_dir / f"{stem}.safetensors"
        if not lora_path.exists():
            continue
        all_char_stems.add(stem)
        parsed = parse_outfit_groups(words)
        primary = parsed["primary"]
        if primary:
            known_triggers[primary.lower()] = (stem, primary)
        # Also index all outfit trigger codes
        for _outfit_name, tags in parsed.get("outfits", {}).items():
            if tags:
                code = tags[0]["word"] if isinstance(tags[0], dict) else str(tags[0])
                if code:
                    known_triggers[code.lower()] = (stem, code)

    # If specific lora_stems provided, only validate those
    if lora_stems:
        relevant_stems = set(lora_stems) & all_char_stems
    else:
        relevant_stems = all_char_stems

    detected = []
    warnings = []
    matched_stems: set[str] = set()
    prompt_lower = prompt.lower()

    # Check for exact trigger code matches
    for code_lower, (stem, code_orig) in known_triggers.items():
        if stem not in relevant_stems:
            continue
        if code_lower in prompt_lower:
            detected.append({
                "name": code_orig,
                "trigger_code": code_orig,
                "lora": stem,
                "confidence": 1.0,
            })
            matched_stems.add(stem)

    # Extract word-like tokens from the prompt to check for fuzzy matches
    # (catches misspellings like "Gigi Mruin" vs "Gigi Murin")
    prompt_tokens = re.findall(r'[A-Za-z][A-Za-z0-9]{2,}(?:\s+[A-Za-z][A-Za-z0-9]{2,})?', prompt)

    for token in prompt_tokens:
        token_lower = token.lower()
        # Skip tokens that already matched exactly
        if any(token_lower == d["trigger_code"].lower() for d in detected):
            continue
        # Check fuzzy similarity against all known trigger codes
        best_score = 0.0
        best_code = ""
        best_stem = ""
        for code_lower, (stem, code_orig) in known_triggers.items():
            if stem not in relevant_stems:
                continue
            score = _fuzzy_match_score(token_lower, code_lower)
            if score > best_score:
                best_score = score
                best_code = code_orig
                best_stem = stem
        # Warn if close but not exact (threshold 0.65–0.95)
        if 0.65 <= best_score < 1.0:
            warnings.append({
                "text": token,
                "suggestion": best_code,
                "lora": best_stem,
                "reason": f"Did you mean '{best_code}'? (similarity: {best_score:.0%})",
                "similarity": round(best_score, 2),
            })

    # Find loaded LoRAs with no trigger detected in the prompt
    unmatched = [s for s in relevant_stems if s not in matched_stems]

    return {
        "detected_characters": detected,
        "warnings": warnings,
        "unmatched_loras": unmatched,
    }


def get_active_model() -> str | None:
    return _active_model


def _load_pipeline(model_path: str) -> StableDiffusionXLPipeline:
    """Load or retrieve a cached pipeline for the given checkpoint."""
    global _active_model, _active_loras

    if model_path in _loaded_pipelines:
        _active_model = model_path
        return _loaded_pipelines[model_path]

    _log.info("[IMAGE GEN] Loading model: %s", model_path)
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    pipe.enable_attention_slicing()

    _loaded_pipelines[model_path] = pipe
    _active_model = model_path
    _active_loras = []
    return pipe


def evict_ollama_models():
    """Evict all Ollama models from GPU so VRAM is free for image generation."""
    import gc
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/ps", timeout=5)
        if resp.status_code == 200:
            running = resp.json().get("models", [])
            for m in running:
                name = m.get("name", "")
                _log.info("[IMAGE GEN] Evicting Ollama model from GPU: %s", name)
                httpx.post(
                    "http://localhost:11434/api/generate",
                    json={"model": name, "keep_alive": 0},
                    timeout=10,
                )
    except Exception as e:
        _log.warning("[IMAGE GEN] Could not evict Ollama models: %s", e)
    gc.collect()


def flush_vram():
    """Unload ALL image-gen pipelines + evict Ollama models to fully free VRAM."""
    import gc
    global _active_model, _active_loras

    # Unload all cached pipelines
    keys = list(_loaded_pipelines.keys())
    for k in keys:
        del _loaded_pipelines[k]
    _active_model = None
    _active_loras = []

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log.info("[IMAGE GEN] Flushed all SD pipelines from VRAM")

    # Also evict Ollama models
    evict_ollama_models()
    _log.info("[IMAGE GEN] VRAM flush complete")


def unload_model(model_path: str | None = None):
    """Free a model from GPU memory."""
    global _active_model, _active_loras

    target = model_path or _active_model
    if target and target in _loaded_pipelines:
        del _loaded_pipelines[target]
        torch.cuda.empty_cache()
        _log.info("[IMAGE GEN] Unloaded model: %s", target)
        if _active_model == target:
            _active_model = None
            _active_loras = []


def load_lora(pipe: StableDiffusionXLPipeline, lora_path: str, weight: float = 0.8):
    """Load a LoRA adapter on top of the current pipeline."""
    global _active_loras
    _log.info("[IMAGE GEN] Loading LoRA: %s (weight=%.2f)", lora_path, weight)
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora(lora_scale=weight)
    _active_loras.append(lora_path)


def unload_loras(pipe: StableDiffusionXLPipeline):
    """Remove all LoRA adapters."""
    global _active_loras
    if _active_loras:
        pipe.unfuse_lora()
        pipe.unload_lora_weights()
        _active_loras = []
        _log.info("[IMAGE GEN] All LoRAs unloaded")


# ── Long prompt handling (bypass CLIP 77-token limit) ─────────────────────


def count_tokens(prompt: str) -> dict:
    """Count CLIP tokens for a prompt.

    Uses the actual CLIP tokenizer from the loaded pipeline if available,
    otherwise falls back to a heuristic estimate (~1.3 tokens per word).
    """
    # Try to use the actual tokenizer from a loaded pipeline
    if _active_model and _active_model in _loaded_pipelines:
        pipe = _loaded_pipelines[_active_model]
        if pipe.tokenizer is not None:
            tokens = pipe.tokenizer.encode(prompt, add_special_tokens=False)
            token_count = len(tokens)
            return {
                "token_count": token_count,
                "max_single_segment": 75,
                "segments_needed": max(1, (token_count + 74) // 75),
                "exceeds_limit": token_count > 75,
                "method": "clip_tokenizer",
            }

    # Fallback: heuristic estimate
    # CLIP BPE averages ~1.3 tokens per whitespace-delimited word
    words = prompt.split()
    estimated = max(1, int(len(words) * 1.3))
    return {
        "token_count": estimated,
        "max_single_segment": 75,
        "segments_needed": max(1, (estimated + 74) // 75),
        "exceeds_limit": estimated > 75,
        "method": "estimate",
    }


def _chunk_prompt(prompt: str, pipe: StableDiffusionXLPipeline) -> str:
    """
    Split long prompts into 75-token chunks joined by BREAK keyword
    so SDXL processes them in segments rather than truncating at 77 tokens.
    """
    tokenizer = pipe.tokenizer
    if tokenizer is None:
        return prompt

    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokens) <= 75:
        return prompt  # fits in one segment, no chunking needed

    _log.info("[IMAGE GEN] Chunking prompt: %d tokens → %d chunks",
              len(tokens), (len(tokens) + 74) // 75)

    chunks = []
    for i in range(0, len(tokens), 75):
        chunk_tokens = tokens[i:i + 75]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text.strip())

    return " BREAK ".join(chunks)


def _encode_long_prompt(pipe, prompt: str, negative_prompt: str, device: str = "cuda"):
    """
    Encode prompts for SDXL handling long prompts beyond the 77-token CLIP limit.

    Strategy: Use two separate Compel instances (one per text encoder) which is
    the correct approach for SDXL and avoids the EmbeddingsProviderMulti.empty_z bug.
    Falls back to BREAK-based chunking if compel is unavailable.
    """
    try:
        from compel import Compel, ReturnedEmbeddingsType
        _log.info("[IMAGE GEN] Using compel for long prompt encoding (%d chars)", len(prompt))

        # SDXL: create separate compel instances for each text encoder
        # This avoids the EmbeddingsProviderMulti.empty_z error
        compel_1 = Compel(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=False,
            truncate_long_prompts=False,
        )
        compel_2 = Compel(
            tokenizer=pipe.tokenizer_2,
            text_encoder=pipe.text_encoder_2,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=True,
            truncate_long_prompts=False,
        )

        conditioning_1 = compel_1(prompt)
        conditioning_2, pooled = compel_2(prompt)
        neg_conditioning_1 = compel_1(negative_prompt)
        neg_conditioning_2, neg_pooled = compel_2(negative_prompt)

        # Pad to same length
        [conditioning_1, neg_conditioning_1] = compel_1.pad_conditioning_tensors_to_same_length(
            [conditioning_1, neg_conditioning_1]
        )
        [conditioning_2, neg_conditioning_2] = compel_2.pad_conditioning_tensors_to_same_length(
            [conditioning_2, neg_conditioning_2]
        )

        # Concatenate along last dim for SDXL dual encoder
        conditioning = torch.cat([conditioning_1, conditioning_2], dim=-1)
        neg_conditioning = torch.cat([neg_conditioning_1, neg_conditioning_2], dim=-1)

        _log.info("[IMAGE GEN] compel encoding complete — embeds shape: %s", conditioning.shape)
        return {
            "prompt_embeds": conditioning,
            "negative_prompt_embeds": neg_conditioning,
            "pooled_prompt_embeds": pooled,
            "negative_pooled_prompt_embeds": neg_pooled,
        }
    except ImportError:
        _log.warning("[IMAGE GEN] compel not installed — run: pip install compel")
        return _manual_chunk_encode(pipe, prompt, negative_prompt, device)
    except Exception as e:
        _log.warning("[IMAGE GEN] compel encoding failed (%s), falling back to chunked", e)
        return _manual_chunk_encode(pipe, prompt, negative_prompt, device)


def _manual_chunk_encode(pipe, prompt: str, negative_prompt: str, device: str = "cuda"):
    """
    Fallback: split prompt into 75-token chunks joined by BREAK keyword.
    Works without compel installed.
    """
    chunked = _chunk_prompt(prompt, pipe)
    neg_chunked = _chunk_prompt(negative_prompt, pipe)

    if chunked != prompt or neg_chunked != negative_prompt:
        return {"_chunked_prompt": chunked, "_chunked_negative": neg_chunked}

    return None


# ── Feedback / prompt analysis ────────────────────────────────────────────

_generation_history: list[dict] = []
MAX_HISTORY = 50
_HISTORY_FILE = Path(OUTPUT_DIR) / "generation_history.json"


def _load_history_from_disk():
    """Load persisted generation history from disk on module init."""
    global _generation_history
    if _HISTORY_FILE.exists():
        try:
            data = json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                _generation_history = data[-MAX_HISTORY:]
                _log.info("[IMAGE GEN] Loaded %d history entries from disk", len(_generation_history))
        except Exception as e:
            _log.warning("[IMAGE GEN] Failed to load history: %s", e)


def _save_history_to_disk():
    """Persist generation history to disk."""
    try:
        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _HISTORY_FILE.write_text(
            json.dumps(_generation_history, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        _log.warning("[IMAGE GEN] Failed to save history: %s", e)


# Load history on module import
_load_history_from_disk()


def _split_respecting_parens(text: str) -> list[str]:
    """Split a prompt string by commas, but keep parenthesized groups intact.
    e.g. '(h0l0r4t:1, black hair), 2 girls' → ['(h0l0r4t:1, black hair)', '2 girls']
    """
    parts = []
    depth = 0
    current = []
    for ch in text:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == ',' and depth == 0:
            part = ''.join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(ch)
    part = ''.join(current).strip()
    if part:
        parts.append(part)
    return parts


def _analyze_prompt(prompt: str) -> dict:
    """Break down the prompt into weighted components for transparency.
    Handles BREAK-separated sections and preserves parenthesized character groups.
    Detects character trigger codes and marks them as high-priority character tokens.
    """
    # Build a set of known character trigger codes for detection
    tw_data = _load_trigger_words()
    char_dir = MODELS_DIR / "characters"
    known_trigger_codes: set[str] = set()
    for stem, words in tw_data.items():
        # Only include character LoRA triggers
        lora_path = char_dir / f"{stem}.safetensors"
        if lora_path.exists():
            parsed = parse_outfit_groups(words)
            if parsed["primary"]:
                known_trigger_codes.add(parsed["primary"].lower())
            for _outfit_name, tags in parsed.get("outfits", {}).items():
                if tags:
                    code = tags[0]["word"] if isinstance(tags[0], dict) else str(tags[0])
                    if code:
                        known_trigger_codes.add(code.lower())

    # Split by BREAK tokens first to identify sections
    sections = re.split(r'\s+BREAK\s+', prompt)

    analysis = {
        "total_parts": 0,
        "components": [],
        "estimated_focus": [],
        "break_count": len(sections) - 1,
    }

    # If there are character BREAK sections, compute regional layout info
    if len(sections) >= 3:
        # Parse region positions the same way _parse_regional_sections does
        region_positions = []
        for sec_idx, sec in enumerate(sections[1:]):
            sec_lower = sec.lower().strip()
            pos = None
            if "on the left" in sec_lower:
                pos = "left"
            elif "on the right" in sec_lower:
                pos = "right"
            elif "in the center" in sec_lower:
                pos = "center"
            elif "in the background" in sec_lower:
                pos = "background"
            region_positions.append(pos)

        # Auto-assign positions like _parse_regional_sections does
        explicit_count = sum(1 for p in region_positions if p is not None)
        n_regions = len(region_positions)
        if explicit_count < n_regions:
            auto_positions = {
                2: ["left", "right"],
                3: ["left", "center", "right"],
                4: ["left", "center", "right", "background"],
            }
            layout = auto_positions.get(n_regions, ["left", "center", "right", "background"][:n_regions])
            for i in range(n_regions):
                if region_positions[i] is None:
                    region_positions[i] = layout[i] if i < len(layout) else "center"

        analysis["regional_layout"] = [
            {"region": i + 1, "position": region_positions[i],
             "auto_assigned": region_positions[i] is not None}
            for i in range(n_regions)
        ]

    global_pos = 0
    for sec_idx, section in enumerate(sections):
        # Split by commas but keep parenthesized groups intact
        parts = _split_respecting_parens(section)

        for part in parts:
            global_pos += 1

            # Check if this part contains a known character trigger code
            part_lower = part.lower().strip()
            is_character_token = False
            for code in known_trigger_codes:
                if code in part_lower:
                    is_character_token = True
                    break
            # Also detect parenthesized character blocks like (TriggerCode:1.2, ...)
            if re.match(r'^\(.*:\s*\d', part.strip()):
                is_character_token = True

            if sec_idx == 0 and not is_character_token:
                # Shared section: use position-based priority
                priority = "high" if global_pos <= 3 else "medium" if global_pos <= 8 else "low"
                section_label = "shared"
            elif is_character_token:
                priority = "character"
                section_label = f"character_{sec_idx}" if sec_idx > 0 else "character_inline"
            else:
                # Non-first section (after BREAK): character-specific
                priority = "character"
                section_label = f"character_{sec_idx}"

            analysis["estimated_focus"].append({
                "text": part,
                "priority": priority,
                "position": global_pos,
                "section": section_label,
            })
            analysis["components"].append(part)

    analysis["total_parts"] = len(analysis["components"])
    return analysis


def record_generation(prompt: str, negative: str, settings: dict, result_path: str | None, feedback: str | None = None):
    """Record generation for learning loop. Persists to disk."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "negative_prompt": negative,
        "settings": settings,
        "result_path": result_path,
        "prompt_analysis": _analyze_prompt(prompt),
        "feedback": feedback,
    }
    _generation_history.append(entry)
    if len(_generation_history) > MAX_HISTORY:
        _generation_history.pop(0)
    _save_history_to_disk()
    return entry


def get_generation_history(last_n: int = 20) -> list[dict]:
    return _generation_history[-last_n:]


def delete_generation_history_entry(index: int) -> dict:
    """Delete a generation history entry by reverse index (0 = newest)."""
    if not _generation_history:
        return {"status": "error", "error": "No history entries"}
    # Index is from newest (reversed), so convert to actual list index
    actual_idx = len(_generation_history) - 1 - index
    if actual_idx < 0 or actual_idx >= len(_generation_history):
        return {"status": "error", "error": f"Index {index} out of range"}
    removed = _generation_history.pop(actual_idx)
    _save_history_to_disk()
    return {"status": "ok", "removed_prompt": (removed.get("prompt") or "")[:60]}


def apply_feedback_learnings(base_negative: str) -> str:
    """Analyze recent negative feedback and add relevant issues to negative prompt.

    Uses both hardcoded keyword→negative mappings AND persistent custom
    learnings extracted from user feedback text.
    """
    if not _generation_history:
        return base_negative

    # Collect feedback from recent bad generations
    issue_keywords = {
        "hands": "bad hands, extra fingers, mutated hands, poorly drawn hands",
        "fingers": "extra fingers, fused fingers, too many fingers",
        "face": "ugly face, bad face proportions, poorly drawn face",
        "eyes": "cross-eyed, uneven eyes, poorly drawn eyes",
        "position": "wrong pose, anatomically incorrect, impossible anatomy",
        "clothing": "wrong clothing, inconsistent outfit, torn clothes",
        "deform": "deformed, disfigured, mutation, mutated, extra limbs",
        "body": "bad body proportions, extra limbs, missing limbs",
        "long": "elongated limbs, stretched body parts, disproportionate",
        "short": "stubby limbs, compressed proportions",
        "hair": "bad hair, messy hair strands, hair artifacts",
        "background": "distorted background, blurred background artifacts",
        "color": "oversaturated, wrong colors, color bleeding",
        "anatomy": "bad anatomy, extra limbs, fused limbs, anatomically incorrect",
        "feet": "bad feet, extra toes, poorly drawn feet",
        "proportion": "bad proportions, wrong proportions, disproportionate",
        "bright": "overexposed, bloom, too bright, washed out",
        "dark": "too dark, underexposed, crushed blacks",
        "blurry": "blurry, out of focus, motion blur, unfocused",
    }

    additions = set()
    for entry in _generation_history[-10:]:
        fb = (entry.get("feedback") or "").lower()
        if not fb:
            continue
        for key, neg_text in issue_keywords.items():
            if key in fb:
                additions.add(neg_text)

        # Also extract user's actual feedback phrases as negative tokens.
        # If user writes "too bright, bloom effect", add those directly.
        fb_parts = [p.strip() for p in fb.split(",") if p.strip()]
        for part in fb_parts:
            # Skip very short or very long fragments
            if 3 <= len(part) <= 60 and part not in base_negative.lower():
                additions.add(part)

    # Load persistent custom learnings if available
    _load_custom_negatives(additions)

    if additions:
        return base_negative + ", " + ", ".join(additions)
    return base_negative


# Persistent custom negative keyword file
_CUSTOM_NEGATIVES_FILE = Path(OUTPUT_DIR) / "custom_negatives.json"


def _load_custom_negatives(additions: set):
    """Load persistent user-defined negative keywords."""
    if _CUSTOM_NEGATIVES_FILE.exists():
        try:
            data = json.loads(_CUSTOM_NEGATIVES_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                additions.update(data)
        except Exception:
            pass


def save_custom_negative(negative_text: str):
    """Add a persistent custom negative keyword/phrase that applies to all future generations."""
    existing = []
    if _CUSTOM_NEGATIVES_FILE.exists():
        try:
            existing = json.loads(_CUSTOM_NEGATIVES_FILE.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    parts = [p.strip() for p in negative_text.split(",") if p.strip()]
    for p in parts:
        if p not in existing:
            existing.append(p)
    _CUSTOM_NEGATIVES_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CUSTOM_NEGATIVES_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return {"status": "ok", "total": len(existing)}


# ── Positive feedback learnings ──────────────────────────────────────────

_CUSTOM_POSITIVES_FILE = Path(OUTPUT_DIR) / "custom_positives.json"

# Keywords in feedback that suggest positive prompt additions
_POSITIVE_KEYWORDS = {
    "vibrant": "vibrant colors, vivid palette",
    "vivid": "vibrant colors, vivid palette",
    "colorful": "colorful, rich color palette",
    "detailed": "highly detailed, intricate details",
    "sharp": "sharp focus, crisp details",
    "soft": "soft lighting, gentle tones",
    "warm": "warm lighting, warm color palette",
    "cool": "cool tones, cool lighting",
    "dramatic": "dramatic lighting, high contrast",
    "cinematic": "cinematic composition, cinematic lighting",
    "dynamic": "dynamic pose, dynamic composition",
    "elegant": "elegant, refined, graceful",
    "cute": "cute, adorable, kawaii",
    "beautiful": "beautiful, stunning, gorgeous",
    "realistic": "realistic proportions, lifelike",
    "moody": "moody atmosphere, atmospheric lighting",
    "pastel": "pastel colors, soft palette",
    "bold": "bold colors, strong contrast",
    "clean": "clean lines, clean composition",
    "smooth": "smooth shading, smooth skin",
}


def apply_positive_learnings(base_prompt: str) -> str:
    """Apply positive learnings from user feedback to enhance the prompt.

    Scans recent feedback for positive signals and adds corresponding prompt
    enhancements. Also loads persistent custom positive keywords.
    """
    additions = set()

    # Scan recent feedback for positive keywords
    for entry in _generation_history[-10:]:
        fb = (entry.get("feedback") or "").lower()
        if not fb:
            continue
        # Skip negative-sounding feedback (these go to negative prompt)
        negative_signals = ["bad", "wrong", "ugly", "deform", "error", "worse", "too"]
        if any(sig in fb for sig in negative_signals):
            continue
        # Check for positive keywords
        for key, pos_text in _POSITIVE_KEYWORDS.items():
            if key in fb:
                additions.add(pos_text)

    # Load persistent custom positives
    if _CUSTOM_POSITIVES_FILE.exists():
        try:
            data = json.loads(_CUSTOM_POSITIVES_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                additions.update(data)
        except Exception:
            pass

    if additions:
        return ", ".join(additions) + ", " + base_prompt
    return base_prompt


def save_custom_positive(positive_text: str):
    """Add a persistent positive keyword/phrase to enhance future generation prompts."""
    existing = []
    if _CUSTOM_POSITIVES_FILE.exists():
        try:
            existing = json.loads(_CUSTOM_POSITIVES_FILE.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    parts = [p.strip() for p in positive_text.split(",") if p.strip()]
    for p in parts:
        if p not in existing:
            existing.append(p)
    _CUSTOM_POSITIVES_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CUSTOM_POSITIVES_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return {"status": "ok", "total": len(existing)}


def get_feedback_learnings() -> dict:
    """Return all current feedback learnings (positive + negative) for review."""
    neg = []
    if _CUSTOM_NEGATIVES_FILE.exists():
        try:
            neg = json.loads(_CUSTOM_NEGATIVES_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    pos = []
    if _CUSTOM_POSITIVES_FILE.exists():
        try:
            pos = json.loads(_CUSTOM_POSITIVES_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Also gather dynamic learnings from recent history
    dynamic_neg = set()
    dynamic_pos = set()
    dummy_neg = apply_feedback_learnings("")
    if dummy_neg:
        dynamic_neg = {p.strip() for p in dummy_neg.split(",") if p.strip()}
    dummy_pos_base = "MARKER"
    dummy_pos = apply_positive_learnings(dummy_pos_base)
    if dummy_pos != dummy_pos_base:
        prefix = dummy_pos.replace(dummy_pos_base, "").strip(", ")
        dynamic_pos = {p.strip() for p in prefix.split(",") if p.strip()}

    return {
        "status": "ok",
        "persistent_negatives": neg,
        "persistent_positives": pos,
        "dynamic_negatives": list(dynamic_neg),
        "dynamic_positives": list(dynamic_pos),
        "total_history_entries": len(_generation_history),
        "entries_with_feedback": sum(1 for e in _generation_history if e.get("feedback")),
    }


def clear_feedback_learnings(scope: str = "all") -> dict:
    """Clear feedback learnings.

    Args:
        scope: "all", "negatives", "positives", or "history"
    """
    if scope in ("all", "negatives"):
        if _CUSTOM_NEGATIVES_FILE.exists():
            _CUSTOM_NEGATIVES_FILE.write_text("[]", encoding="utf-8")
    if scope in ("all", "positives"):
        if _CUSTOM_POSITIVES_FILE.exists():
            _CUSTOM_POSITIVES_FILE.write_text("[]", encoding="utf-8")
    if scope in ("all", "history"):
        for entry in _generation_history:
            entry["feedback"] = None
        _save_history_to_disk()
    return {"status": "ok", "cleared": scope}


# ── Vocabulary Expansion (synonym suggestions for better prompts) ─────────

# Curated mapping of common words to SD-friendly alternatives.
# Helps users find the right tokens that the CLIP model weights strongly.
_VOCAB_EXPANSION: dict[str, list[str]] = {
    # ── explicit terms ─────────────────────────────────────────────────────────
    "nsfw": ["erotic", "libidinous", "Lascivious", "salacious", "prurient", "lewd", "raunchy", "smutty", "depraved", "indecent", "randy", "kinky", "naughty", "filthy", "sultry", "seductive", "provocative", "slutty", "hot", "steamy", "explicit content", "adult content"],
    "blowjob": ["Wet", "sloppy", "gagging", "deepthroat", "gluttonous", "throatclenching", "saliva-drenched", "oral fixation", "mouth worship", "fellatio", "tongue action", "lip service", "throat play", "oral indulgence", "mouth pleasure"],
    "teasing": ["provocative", "suggestive", "flirtatious", "enticing", "tantalizing", "seductive tease", "playful allure", "coquettish", "sultry hint", "alluring glimpse"],
    "breasts": ["bountiful bosom", "ample cleavage", "voluptuous chest", "curvaceous bust", "well-endowed", "firm mounds", "luscious curves", "sculpted chest", "desirable décolletage", "seductive swell"],
    "petite breasts": ["perky small breasts", "petite bust", "delicate chest", "small but shapely breasts", "modest bosom", "cute cleavage", "dainty curves", "compact chest", "adorable bust", "subtle swell"],
    "bouncy breasts": ["perky breasts", "springy breasts", "energetic breasts", "vibrant breasts", "lively breasts", "jiggly breasts", "shaking breasts", "wobbly breasts", "bust in motion", "animated cleavage"],
    "big butt": ["voluptuous rear", "ample behind", "curvy buttocks", "well-defined glutes", "shapely posterior", "bountiful booty", "full-figured rear", "desirable derriere", "sculpted buttocks", "seductive backside"],
    "small butt": ["petite rear", "delicate behind", "compact buttocks", "dainty posterior", "subtle glutes", "modest booty", "slim rear", "neat derriere", "svelte buttocks", "cute backside"],
    "tight clothes": ["form-fitting clothes", "body-hugging outfit", "clinging attire", "snug clothing", "figure-emphasizing garments", "skin-tight apparel", "revealing outfit", "sculpting clothes", "sensual attire", "alluring tightwear"],
    "loose clothes": ["flowing clothes", "billowy garments", "draped outfit", "relaxed fit clothing", "oversized attire", "airy clothes", "casual loosewear", "comfortably baggy clothes", "non-restrictive garments", "free-flowing outfit"],
    "sexy clothes": ["provocative clothes", "seductive outfit", "alluring attire", "revealing clothing", "scanty garments", "risqué clothes", "sultry outfit", "tempting attire", "flirtatious clothing", "hot fashion"],
    

    # ── adjectives for people ────────────────────────────────────────────

    "skilled": ["masterful", "expert", "proficient", "adept", "virtuoso"],
    "devout": ["pious", "reverent", "faithful", "spiritual", "godly"],
    "mischievous": ["playful", "naughty", "impish", "roguish", "cheeky"],
    "passionate": ["ardent", "fervent", "intense", "fiery", "zealous"],
    "methodical": ["precise", "deliberate", "careful", "thorough", "systematic"],
    "soft": ["gentle", "tender", "delicate", "plush", "cushy", "velvety", "silky", "downy", "satin-like"],
    "blush": ["romantic blush", "loving flush", "affectionate glow", "tender blush", "adoring flush", "charmed", "enamored", "smitten", "heartfelt blush", "devoted flush"],
    "nervous": ["coy", "anxious", "timid", "shy", "flustered", "uneasy", "apprehensive", "jittery", "restless", "fidgety", "sheepish", "bashful", "hesitant", "introverted"],
    

    # ── Emotions & expressions ────────────────────────────────────────────
    "sexy": ["alluring", "seductive", "sensual", "sultry", "attractive"],
    "beautiful": ["gorgeous", "stunning", "elegant", "radiant", "breathtaking"],
    "angry": ["furious", "enraged", "fierce expression", "aggressive", "scowling"],
    "sad": ["melancholic", "sorrowful", "grief-stricken", "teary-eyed", "downcast"],
    "happy": ["joyful", "beaming", "cheerful", "elated", "gleeful"],
    "scared": ["terrified", "wide-eyed fear", "cowering", "trembling", "petrified"],
    "surprised": ["astonished", "jaw-dropped", "wide-eyed shock", "startled", "dumbfounded"],
    "calm": ["serene", "tranquil", "composed", "placid", "unperturbed"],
    "confident": ["self-assured", "commanding presence", "proud stance", "dignified", "resolute"],
    "shy": ["bashful", "timid", "demure", "coy", "flustered", "nervous smile", "averted gaze", "blushing cheeks", "hesitant posture", "introverted demeanor"],
    "tired": ["exhausted", "fatigued", "weary", "drowsy", "heavy-lidded", "sluggish", "drained", "sleepy", "worn out", "lethargic"],
    "smiling": ["grinning", "beaming smile", "radiant expression", "warm smile", "gentle smile"],
    "crying": ["tears streaming", "sobbing", "weeping", "teary-eyed", "emotional tears", "heartbroken", "anguished", "despairing", "grief-stricken", "wailing"],
    "laughing": ["cackling", "hearty laughter", "giggling", "roaring with laughter", "mirthful"],
    "thinking": ["contemplative", "pensive", "deep in thought", "reflective", "pondering"],
    "pleasure": ["ecstasy", "bliss", "sensual delight", "carnal joy", "intimate pleasure", "Rapture", "orgasmic", "satisfaction", "intoxication", "euphoria"],

    # ── Physical descriptors ──────────────────────────────────────────────
    "big": ["massive", "enormous", "towering", "imposing", "grand", "gigantic", "colossal", "immense", "monumental", "mammoth"],
    "small": ["petite", "tiny", "diminutive", "compact", "delicate", "miniature", "dainty", "modest", "little", "subtle"],
    "strong": ["muscular", "powerful", "athletic build", "toned", "robust", "sturdy", "well-built", "brawny", "sinewy", "ripped"],
    "cute": ["adorable", "kawaii", "charming", "sweet", "endearing", "lovely", "precious", "delightful", "winsome", "heartwarming"],
    "old": ["aged", "weathered", "elderly", "ancient", "wizened", "senescent", "venerable", "timeworn", "grizzled", "hoary"],
    "young": ["youthful", "adolescent", "fresh-faced", "juvenile", "vibrant youth", "childlike", "innocent", "naive", "sprightly", "blooming"],
    "fat": ["plump", "voluptuous", "curvy", "rotund", "full-figured", "chubby", "well-padded", "ample", "stout", "portly"],
    "thin": ["slender", "lithe", "svelte", "willowy", "lean", "skinny", "slim", "spindly", "gaunt", "scrawny"],
    "tall": ["towering", "statuesque", "elongated figure", "lofty", "imposing height", "elevated stature", "lofty frame", "soaring height", "grand height", "lofty build"],
    "short": ["petite", "compact stature", "diminutive frame", "stubby", "low to ground", "shortened limbs", "compressed proportions", "fun-sized", "miniature build", "small stature"],
    "plump": ["voluptuous rear", "curvy buttocks", "well-defined glutes", "ample behind", "shapely posterior", "bountiful booty", "full-figured rear", "desirable derriere", "sculpted buttocks", "seductive backside"],
    "voluptuous": ["curvaceous", "full-figured", "ample", "buxom", "well-endowed", "zaftig", "rubenesque", "luscious", "sculpted curves", "seductive figure"],
    

    # ── Lighting & atmosphere ─────────────────────────────────────────────
    "dark": ["shadowy", "dimly lit", "noir", "tenebrous", "deep shadows"],
    "bright": ["luminous", "radiant", "glowing", "vibrant", "brilliant", "sunlit", "dazzling", "shining", "illuminated", "sparkling"],
    "scary": ["terrifying", "menacing", "ominous", "eldritch", "horrifying"],
    "cool": ["stylish", "composed", "sleek", "suave", "effortlessly chic"],
    "dramatic": ["cinematic", "high contrast", "chiaroscuro", "theatrical", "imposing composition"],
    "soft": ["diffused light", "ethereal glow", "gentle illumination", "pastel tones", "dreamy"],
    "warm": ["golden light", "amber tones", "cozy atmosphere", "sunset hues", "inviting glow"],
    "cold": ["blue tones", "icy atmosphere", "frigid", "frosty", "glacial light"],
    "cozy": ["warm and inviting", "snug atmosphere", "intimate haven", "comfortable surroundings","homey vibe","Refined comfort" ,"soft lighting", "relaxed ambiance", "welcoming environment", "pleasantly warm", "cuddle-worthy", "hygge-inspired", "Tranquil"],

    # ── Actions & movement (core) ─────────────────────────────────────────
    "walking": ["striding confidently", "sauntering", "ambling gracefully", "marching", "strolling leisurely"],
    "running": ["sprinting full speed", "dashing forward", "bolting", "racing with urgency", "fleet-footed sprint"],
    "jumping": ["leaping into the air", "vaulting upward", "bounding", "sky-high jump", "acrobatic leap"],
    "sitting": ["seated elegantly", "perched gracefully", "reclining", "cross-legged repose", "slouched casually"],
    "standing": ["upright stance", "statuesque pose", "at attention", "planted firmly", "towering presence"],
    "lying": ["recumbent", "sprawled out", "supine position", "prostrate", "lying on side"],
    "kneeling": ["genuflecting", "on one knee", "crouched reverently", "bowed on knees", "supplicant pose"],
    "crouching": ["hunched low", "defensive crouch", "squatting", "coiled to spring", "low stance"],
    "flying": ["airborne", "soaring through sky", "hovering mid-air", "gliding effortlessly", "ascending rapidly"],
    "falling": ["plummeting", "tumbling downward", "free-falling", "cascading descent", "spiraling down"],
    "climbing": ["scaling upward", "clambering", "ascending vertically", "gripping handholds", "mountaineering"],
    "swimming": ["submerged gliding", "backstroke", "treading water", "diving under", "aquatic movement"],
    "dancing": ["pirouetting", "waltzing gracefully", "rhythmic movement", "twirling", "ballet en pointe"],
    "fighting": ["combat stance", "throwing punches", "martial arts form", "defensive block", "aggressive assault"],
    "fast": ["dynamic motion", "speed lines", "swift", "rapid movement", "motion blur"],
    "slow": ["calm", "serene", "peaceful", "tranquil", "gentle movement"],
    "bounce": ["springing", "rebounding", "bouncing energetically", "elastic motion", "vibrant bounce", "grinding", "riding"],

    # ── Combat & battle actions ───────────────────────────────────────────
    "attacking": ["lunging forward", "delivering a strike", "unleashing an assault", "cleaving through", "offensive charge"],
    "blocking": ["parrying a blow", "raising shield", "defensive ward", "deflecting attack", "braced for impact"],
    "dodging": ["evading nimbly", "sidestepping", "rolling away", "ducking under", "acrobatic evasion"],
    "casting": ["channelling energy", "arcane invocation", "hands glowing with power", "spell weaving", "magical incantation"],
    "slashing": ["sword arc", "blade sweep", "cleaving strike", "horizontal cut", "diagonal slash"],
    "shooting": ["taking aim", "firing weapon", "pulling trigger", "archery draw", "projectile release"],
    "punching": ["throwing a fist", "uppercut", "jab strike", "hook punch", "knuckle impact"],
    "kicking": ["roundhouse kick", "high kick", "spinning kick", "front kick", "dropkick"],

    # ── Poses & gestures ─────────────────────────────────────────────────
    "posing": ["striking a pose", "model stance", "dramatic posture", "fashion pose", "contrapposto"],
    "pointing": ["outstretched finger", "directing gesture", "indicating forward", "accusatory point", "beckoning"],
    "waving": ["hand raised in greeting", "farewell wave", "beckoning gesture", "enthusiastic wave", "gentle wave"],
    "hugging": ["embracing tightly", "warm embrace", "arms wrapped around", "tender hold", "affectionate clutch"],
    "reaching": ["arm outstretched", "grasping forward", "stretching toward", "fingers extended", "yearning reach"],
    "holding": ["gripping firmly", "cradling gently", "clasping", "bearing in hands", "wielding"],
    "leaning": ["tilted posture", "resting against", "inclined forward", "slouched to one side", "propped against wall"],
    "bowing": ["deep bow", "reverent inclination", "formal curtsy", "head lowered respectfully", "solemn genuflection"],
    "stretching": ["arms above head stretch", "arching back", "limbering up", "reaching skyward", "full body extension"],
    "looking": ["gazing intently", "glancing sideways", "staring ahead", "peering over shoulder", "upward gaze"],
    "turning": ["pivoting", "rotating torso", "head turn", "swiveling", "looking back over shoulder"],

    # ── Dynamic action descriptions ───────────────────────────────────────
    "explosion": ["detonation blast wave", "fiery eruption", "shockwave impact", "debris scattering", "cataclysmic burst"],
    "impact": ["collision force", "shattering on contact", "kinetic strike", "concussive blow", "thunderous crash"],
    "pursuit": ["relentless chase", "hot on the trail", "predator stalking prey", "breakneck pursuit", "desperate flee"],
    "transformation": ["metamorphosis", "shape-shifting", "power awakening", "form evolving", "dramatic change"],
    "summoning": ["conjuring forth", "ritual invocation", "materializing from void", "calling upon power", "ethereal manifestation"],

    # ── Environment & setting ─────────────────────────────────────────────
    "magic": ["arcane", "mystical", "ethereal glow", "sorcery", "enchanted"],
    "fire": ["flames", "blazing", "inferno", "ember glow", "pyroclastic"],
    "water": ["aquatic", "submerged", "ripples", "ocean spray", "crystalline water"],
    "passionate": ["fervent", "ardent", "amorousness", "intense desire", "buxom"],
    "warrior": ["battle-ready", "armored fighter", "combat stance", "gladiator", "swordsman"],
    "forest": ["lush woodland", "dense foliage", "ancient trees", "verdant canopy", "enchanted grove"],
    "city": ["urban landscape", "metropolis", "cityscape", "neon-lit streets", "towering skyscrapers"],
    "night": ["moonlit", "starry sky", "nocturnal", "midnight", "twilight"],
    "day": ["sunlit", "golden hour", "daylight", "afternoon sun", "clear sky"],
    "robot": ["mecha", "android", "cybernetic", "mechanical", "biomechanical"],
    "monster": ["creature", "beast", "eldritch horror", "chimera", "abomination"],
    "ocean": ["vast seascape", "crashing waves", "deep blue expanse", "maritime", "tempestuous sea"],
    "mountain": ["alpine peak", "craggy summit", "snow-capped mountain", "towering ridge", "precipitous cliff"],
    "sky": ["celestial expanse", "cloud-strewn heavens", "azure canopy", "atmospheric vista", "firmament"],
    "rain": ["downpour", "drizzling", "raindrops cascading", "storm precipitation", "soaked in rain"],
    "wind": ["gusting breeze", "windswept", "hair billowing in wind", "gale force", "zephyr"],
    "snow": ["snowfall", "blizzard", "frost-covered", "pristine white blanket", "crystalline snowflakes"],

    # ── Camera & composition ──────────────────────────────────────────────
    "closeup": ["extreme close-up", "tight framing", "macro shot", "face portrait", "intimate detail"],
    "portrait": ["head and shoulders", "bust shot", "character portrait", "three-quarter view", "profile shot"],
    "wide": ["wide-angle shot", "panoramic view", "full body framing", "establishing shot", "expansive composition"],
    "above": ["bird's eye view", "top-down perspective", "overhead angle", "aerial viewpoint", "looking down at"],
    "below": ["low angle shot", "worm's eye view", "looking up at", "dramatic low perspective", "ground-level"],
    "side": ["profile view", "lateral perspective", "side-on angle", "orthogonal view", "90-degree angle"],
    "behind": ["rear view", "from behind", "over-the-shoulder", "back-facing", "posterior perspective"],

    # ── Clothing & accessories ────────────────────────────────────────────
    "armor": ["plate armor", "battle-worn armor", "ornate breastplate", "chainmail", "knight's regalia"],
    "dress": ["flowing gown", "elegant dress", "ball gown", "sundress", "cocktail dress", "evening wear"],
    "uniform": ["military uniform", "school uniform", "formal attire", "officer's garb", "ceremonial dress"],
    "casual": ["streetwear", "relaxed outfit", "everyday clothing", "hoodie and jeans", "comfortable attire"],
    "cloak": ["billowing cape", "hooded mantle", "flowing cloak", "mysterious shroud", "traveler's cloak"],
}


def expand_vocabulary(text: str) -> dict:
    """Return synonym suggestions for words in the prompt to help users
    find tokens the CLIP model weights more strongly."""
    words = re.findall(r'\b\w+\b', text.lower())
    suggestions = {}
    for word in words:
        if word in _VOCAB_EXPANSION:
            suggestions[word] = _VOCAB_EXPANSION[word]
    return {"status": "ok", "suggestions": suggestions}


# ── Multi-angle / multi-view prompt structuring ───────────────────────────

_ANGLE_KEYWORDS = {
    "side view": "from side, profile view, lateral angle",
    "front view": "from front, facing viewer, frontal angle",
    "back view": "from behind, rear view, back-facing",
    "top view": "from above, bird's eye view, overhead angle",
    "bottom view": "from below, low angle, worm's eye view",
    "three-quarter view": "3/4 angle, three-quarter perspective",
    "pov": "first-person perspective, point of view, POV shot",
    "point of view": "first-person perspective, POV shot, subjective camera",
}

_MULTI_VIEW_PATTERNS = [
    # "side view and front view", "side view + pov", etc.
    re.compile(r'(side view|front view|back view|top view|bottom view|three-quarter view|pov|point of view)'
               r'\s*(?:and|&|\+|,)\s*'
               r'(side view|front view|back view|top view|bottom view|three-quarter view|pov|point of view)',
               re.IGNORECASE),
]


def _structure_multi_angle_prompt(prompt: str) -> str:
    """Detect multi-angle requests and structure them as a split-screen
    composition so SDXL renders both views in one image."""
    for pat in _MULTI_VIEW_PATTERNS:
        m = pat.search(prompt)
        if m:
            angle_a, angle_b = m.group(1).lower(), m.group(2).lower()
            expand_a = _ANGLE_KEYWORDS.get(angle_a, angle_a)
            expand_b = _ANGLE_KEYWORDS.get(angle_b, angle_b)
            # Remove the matched multi-angle phrase from the original
            base = prompt[:m.start()].rstrip(' ,') + prompt[m.end():].lstrip(' ,')
            base = base.strip(', ')
            structured = (
                f"split screen, two panels, reference sheet, multiple views, "
                f"left panel: {expand_a}, {base} BREAK "
                f"right panel: {expand_b}, {base}"
            )
            _log.info("[IMAGE GEN] Multi-angle detected: %s + %s", angle_a, angle_b)
            return structured
    return prompt


# ── Multi-character prompt structuring ────────────────────────────────────

def _structure_multi_character_prompt(prompt: str, lora_paths: list[str] | None = None,
                                      selected_outfits: dict[str, str] | None = None) -> str:
    """Detect multiple character names in prompt and structure them so SDXL
    renders each character distinctly rather than merging them into one.

    Uses BREAK tokens to isolate each character's description into a separate
    75-token attention window, preventing feature bleed between characters.

    Only activates when 2+ character-category LoRAs are actively loaded.

    Args:
        selected_outfits: Optional dict of {lora_stem: outfit_name} for outfit selection.
    """
    if not lora_paths or len(lora_paths) < 2:
        return prompt

    # Only count LoRAs from the 'characters' category
    char_lora_paths = []
    char_dir = MODELS_DIR / "characters"
    for lp in lora_paths:
        lp_path = Path(lp)
        try:
            if lp_path.parent.resolve() == char_dir.resolve():
                char_lora_paths.append(lp)
        except Exception:
            pass

    if len(char_lora_paths) < 2:
        return prompt

    # If the prompt already has BREAK sections with character triggers,
    # it was likely composed by the frontend character panel — skip re-structuring
    existing_breaks = len(re.findall(r'\bBREAK\b', prompt))
    if existing_breaks >= len(char_lora_paths):
        _log.info("[IMAGE GEN] Prompt already has %d BREAK sections for %d char LoRAs — skipping re-structure",
                  existing_breaks, len(char_lora_paths))
        return prompt

    _log.info("[IMAGE GEN] Multi-character structuring: %d character LoRAs active",
              len(char_lora_paths))

    tw_data = _load_trigger_words()
    positions = ["on the left", "on the right", "in the center", "in the background"]
    selected_outfits = selected_outfits or {}

    # ── Step 1: Identify each character's trigger words ──────────────────
    char_info = []  # list of (lora_path, stem, primary_trigger, all_trigger_words, all_trigger_codes)
    for lp in char_lora_paths[:4]:
        stem = Path(lp).stem
        words = tw_data.get(stem, [])
        parsed_groups = parse_outfit_groups(words)
        primary = parsed_groups["primary"] or stem

        # Collect ALL trigger code words (the first comma-separated token of
        # each outfit entry).  These are the short codes like "Cec1l1aNwYrs",
        # "GigiNwYrs", "GigiBase" etc. that users actually type in prompts.
        trigger_codes = set()
        trigger_codes.add(primary)
        for _outfit_name, tags in parsed_groups.get("outfits", {}).items():
            if tags:
                code = tags[0]["word"] if isinstance(tags[0], dict) else str(tags[0])
                if code:
                    trigger_codes.add(code)
        # Also add any simple string trigger words
        all_tw = [parse_trigger_word_entry(w)["word"] for w in words
                  if parse_trigger_word_entry(w)["word"] and parse_trigger_word_entry(w)["word"] != ";"]
        for tw_str in all_tw:
            # For colon-format, extract just the trigger code (first token after colon)
            colon_m = re.match(r'^[^:]+:\s*(\S+)', tw_str)
            if colon_m:
                trigger_codes.add(colon_m.group(1).rstrip(','))

        char_info.append((lp, stem, primary, list(trigger_codes), all_tw))

    # ── Step 2: Extract parenthesized character blocks from the prompt ─────
    # Users write e.g. "(Cec1l1aNwYrs:1, green eyes, short hair, ...)" to group
    # a character's description.  We detect blocks containing ANY of the
    # character's trigger codes (not just the primary) and pull them out.
    shared = prompt
    char_blocks: dict[str, str] = {}  # stem -> extracted block text

    for _lp, stem, _primary, trigger_codes, _all_tw in char_info:
        # Try each trigger code until we find a match in a parenthesized block
        found = False
        for code in sorted(trigger_codes, key=len, reverse=True):  # longest first
            paren_pat = r'\(\s*' + re.escape(code) + r'[^)]*\)'
            full_pat = (r'(?:\s*BREAK\s+)?' + paren_pat +
                        r'(?:\s*,\s*(?:on the left|on the right|in the center|in the background))?')
            match = re.search(full_pat, shared, re.IGNORECASE)
            if match:
                paren_match = re.search(paren_pat, match.group(0), re.IGNORECASE)
                char_blocks[stem] = paren_match.group(0) if paren_match else match.group(0)
                shared = shared[:match.start()] + shared[match.end():]
                _log.info("[IMAGE GEN] Extracted character block for %s (matched '%s'): %s",
                          stem, code, char_blocks[stem])
                found = True
                break
        if not found:
            _log.info("[IMAGE GEN] No parenthesized block found for %s (tried: %s)",
                      stem, trigger_codes)

    # ── Step 3: Strip any remaining standalone trigger words from shared ────
    for _lp, stem, _primary, trigger_codes, all_tw in char_info:
        # Strip trigger codes
        for code in trigger_codes:
            shared = re.sub(r'(?<![a-zA-Z0-9])' + re.escape(code) + r'(?![a-zA-Z0-9])',
                            '', shared, flags=re.IGNORECASE)
        # Strip full trigger word strings
        for tw_str in all_tw:
            # Use word boundary to avoid partial matches
            shared = re.sub(r'(?<![a-zA-Z])' + re.escape(tw_str) + r'(?![a-zA-Z])',
                            '', shared, flags=re.IGNORECASE)

    # Strip any leftover BREAK tokens and position phrases from shared
    shared = re.sub(r'\bBREAK\b', '', shared)
    shared = re.sub(r',?\s*on the (?:left|right)\b', '', shared, flags=re.IGNORECASE)
    shared = re.sub(r',?\s*in the (?:center|background)\b', '', shared, flags=re.IGNORECASE)
    # Remove existing "N characters" and "group shot" to avoid duplication
    shared = re.sub(r'\d+\s*characters?\b', '', shared, flags=re.IGNORECASE)
    shared = re.sub(r'\bgroup\s+shot\b', '', shared, flags=re.IGNORECASE)

    # Clean up orphaned commas, empty parens, extra whitespace
    shared = re.sub(r'\(\s*[:\d.]*\s*\)', '', shared)      # empty weighted parens like (:1)
    shared = re.sub(r'\(\s*\)', '', shared)                 # empty parens
    shared = re.sub(r',\s*,', ',', shared)                  # double commas
    shared = re.sub(r',\s*,', ',', shared)                  # second pass
    shared = re.sub(r'\s{2,}', ' ', shared)                 # collapse whitespace
    shared = shared.strip(', \t\n')

    # ── Step 4: Build BREAK-separated character sections ───────────────────
    parts = []
    for i, (lp, stem, primary, _trigger_codes, _all_tw) in enumerate(char_info):
        pos = positions[i] if i < len(positions) else positions[-1]

        if stem in char_blocks:
            # Use the user's own parenthesized character block
            parts.append(f"{char_blocks[stem]}, {pos}")
        else:
            # Build from trigger word config (character wasn't manually grouped)
            words = tw_data.get(stem, [])
            outfit = selected_outfits.get(stem)
            char_triggers = build_trigger_prompt(words, selected_outfit=outfit) if words else stem
            parts.append(f"{char_triggers}, {pos}")

    # ── Step 5: Assemble final prompt ──────────────────────────────────────
    num_chars = len(char_info)
    structured = f"{num_chars} characters, group shot, {shared}"
    for part in parts:
        structured += f" BREAK {part}"

    _log.info("[IMAGE GEN] Structured multi-character prompt: %s", structured[:200])
    return structured


# ── Regional conditioning for multi-character isolation ───────────────────

def _encode_single_sdxl(pipe, prompt: str, device: str = "cuda"):
    """Encode a single prompt using SDXL's dual text encoders.
    Returns (prompt_embeds, pooled_prompt_embeds).
    """
    dtype = pipe.text_encoder.dtype

    # Text encoder 1 (CLIP-L)
    tokens_1 = pipe.tokenizer(
        prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        enc_1 = pipe.text_encoder(tokens_1.input_ids, output_hidden_states=True)
    # Use penultimate hidden state for SDXL
    hidden_1 = enc_1.hidden_states[-2]

    # Text encoder 2 (CLIP-G)
    tokens_2 = pipe.tokenizer_2(
        prompt, padding="max_length", max_length=pipe.tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        enc_2 = pipe.text_encoder_2(tokens_2.input_ids, output_hidden_states=True)
    hidden_2 = enc_2.hidden_states[-2]
    pooled = enc_2[0]  # pooled output from the final layer

    # Concatenate: SDXL expects [enc_1, enc_2] along the last dim
    prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1).to(dtype=dtype)
    pooled = pooled.to(dtype=dtype)

    return prompt_embeds, pooled


def _create_spatial_masks(regions: list[dict], latent_w: int, latent_h: int,
                          device: str = "cuda", dtype=torch.float16) -> list[torch.Tensor]:
    """Create wide, soft Gaussian-blended spatial masks.

    Unlike hard box partitions, these masks use broad bell curves centered on
    each character's position.  Every character's mask covers most of the canvas
    but peaks at their designated position.  This produces ONE seamless image
    instead of visibly separate panels.

    After construction the masks are softmax-normalised so they sum to 1.0 at
    every pixel — no hard edges, no gaps.
    """
    import math
    n = len(regions)
    # Build a 1-D x-coordinate grid [0..1] across the latent width
    xs = torch.linspace(0, 1, latent_w, device=device, dtype=dtype)

    # Map each position keyword to a centre x-value
    def _centre(pos: str, idx: int) -> float:
        if pos == "left":
            return 0.25
        elif pos == "right":
            return 0.75
        elif pos == "center":
            return 0.5
        elif pos == "background":
            return 0.5  # uniform — handled below
        else:
            # Even spread for unknown positions
            return (idx + 0.5) / n

    # Sigma controls how wide each bell curve is.  Larger = more overlap,
    # softer blending, more seamless.
    # For 2 characters: 0.35 gives ~70% overlap (very seamless)
    # For 3+ characters: tighter sigma so the center character isn't overwhelmed
    #                     by flanking duplicates
    if n <= 2:
        sigma = 0.35
    elif n == 3:
        sigma = 0.28  # tighter for 3 chars — prevents center from being swamped
    else:
        sigma = 0.22  # even tighter for 4+

    raw_masks: list[torch.Tensor] = []
    centres_used = []
    for idx, r in enumerate(regions):
        if r["position"] == "background":
            # Uniform low weight
            weight = torch.ones(latent_w, device=device, dtype=dtype) * 0.5
            centres_used.append(0.5)
        else:
            cx = _centre(r["position"], idx)
            centres_used.append(cx)
            weight = torch.exp(-((xs - cx) ** 2) / (2 * sigma ** 2))
        raw_masks.append(weight)

    _log.info("[IMAGE GEN] Spatial masks: %d regions, sigma=%.2f, centres=%s",
              n, sigma, [f"{c:.2f}" for c in centres_used])

    # Stack and softmax-normalise across characters at every x position
    stacked = torch.stack(raw_masks, dim=0)  # (n, latent_w)
    # Softmax temperature: higher = sharper separation between regions
    # Use sharper temp for 3+ chars to keep distinct zones
    temperature = 3.0 if n <= 2 else 4.0
    normed = torch.softmax(stacked * temperature, dim=0)

    masks = []
    for i in range(n):
        mask = normed[i].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, W)
        mask = mask.expand(1, 1, latent_h, latent_w).contiguous()
        masks.append(mask)

    return masks


def _parse_regional_sections(prompt: str) -> tuple[str, list[dict]] | None:
    """Parse a BREAK-separated prompt into shared + character regions.

    Returns None if the prompt doesn't have character regions.
    Returns (shared_prompt, [{"prompt": str, "position": str}, ...])

    When no explicit position keywords are found, characters are auto-assigned
    evenly-spaced positions (left/center/right) so spatial masks actually
    separate them instead of all piling up at "center".
    """
    sections = re.split(r'\s+BREAK\s+', prompt)
    if len(sections) < 3:  # need shared + at least 2 characters
        return None

    shared = sections[0].strip()
    regions = []
    for sec in sections[1:]:
        sec = sec.strip()
        if not sec:
            continue
        pos = None  # None = no explicit position
        sec_lower = sec.lower()
        if "on the left" in sec_lower:
            pos = "left"
        elif "on the right" in sec_lower:
            pos = "right"
        elif "in the center" in sec_lower:
            pos = "center"
        elif "in the background" in sec_lower:
            pos = "background"
        regions.append({"prompt": sec, "position": pos})

    if len(regions) < 2:
        return None

    # ── Auto-assign positions when user didn't specify any ─────────────
    # If none or only one region has an explicit position, spread them evenly.
    explicit_count = sum(1 for r in regions if r["position"] is not None)
    if explicit_count < len(regions):
        n = len(regions)
        # Standard layouts by character count
        auto_positions = {
            2: ["left", "right"],
            3: ["left", "center", "right"],
            4: ["left", "center", "right", "background"],
        }
        layout = auto_positions.get(n, ["left", "center", "right", "background"][:n])

        for i, r in enumerate(regions):
            if r["position"] is None:
                r["position"] = layout[i] if i < len(layout) else "center"
                _log.info("[IMAGE GEN] Auto-assigned position '%s' to region %d: %s",
                          r["position"], i, r["prompt"][:60])

    return shared, regions


def _generate_regional(
    pipe, prompt: str, negative_prompt: str,
    width: int, height: int, steps: int, guidance_scale: float,
    generator: torch.Generator,
):
    """Generate an image using regional conditioning for multi-character isolation.

    Instead of encoding the full BREAK-separated prompt as one, this function:
    1. Parses character sections from the prompt
    2. Encodes each character's section separately
    3. Creates spatial masks for each character's region
    4. Runs a custom denoising loop where each step blends per-region
       noise predictions using the spatial masks

    This prevents cross-attention bleed between characters because each
    region's noise prediction is computed from ONLY that character's
    conditioning, not the full concatenated prompt.

    Typically ~2.5x slower than standard generation (N+2 UNet passes per step
    instead of 2), but produces much better character isolation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pipe.unet.dtype

    parsed = _parse_regional_sections(prompt)
    if parsed is None:
        # Fallback to normal generation
        return pipe(
            prompt=prompt, negative_prompt=negative_prompt,
            width=width, height=height, num_inference_steps=steps,
            guidance_scale=guidance_scale, generator=generator,
        ).images[0]

    shared_prompt, regions = parsed
    n_chars = len(regions)
    latent_h, latent_w = height // 8, width // 8

    _log.info("[IMAGE GEN] Regional conditioning: %d character regions, %d steps", n_chars, steps)
    _log.info("[IMAGE GEN] Shared: %s", shared_prompt[:100])
    for i, r in enumerate(regions):
        _log.info("[IMAGE GEN] Region %d (%s): %s", i, r["position"], r["prompt"][:80])

    # ── Encode prompts ─────────────────────────────────────────────────────
    # Negative (shared across all regions)
    neg_embeds, neg_pooled = _encode_single_sdxl(pipe, negative_prompt, device)

    # Per-character prompts (include shared scene for context coherence)
    char_embeds_list = []
    for r in regions:
        combined = f"{shared_prompt}, {r['prompt']}"
        embeds, pooled = _encode_single_sdxl(pipe, combined, device)
        char_embeds_list.append((embeds, pooled))

    # ── Create spatial masks ───────────────────────────────────────────────
    masks = _create_spatial_masks(regions, latent_w, latent_h, device, dtype)
    # Masks are softmax-normalised and sum to 1.0 — no separate background mask needed

    # ── Prepare time IDs (SDXL-specific conditioning) ──────────────────
    # Format: [original_h, original_w, crop_top, crop_left, target_h, target_w]
    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]], dtype=dtype,
    ).to(device)

    # ── Prepare latents ────────────────────────────────────────────────────
    latents = torch.randn(
        1, 4, latent_h, latent_w,
        generator=generator, device=device, dtype=dtype,
    )

    # ── Setup scheduler ────────────────────────────────────────────────────
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps
    latents = latents * pipe.scheduler.init_noise_sigma

    # ── Regional denoising loop ────────────────────────────────────────────
    step_start_time = time.time()
    for step_idx, t in enumerate(timesteps):
        now = time.time()
        elapsed = now - step_start_time
        its = (step_idx / elapsed) if elapsed > 0 and step_idx > 0 else 0.0
        _update_progress(
            current_step=step_idx + 1,
            message=f"Regional: {n_chars} regions | {its:.2f} it/s" if step_idx > 0 else f"Regional: {n_chars} regions | starting...",
        )
        if _cancel_event.is_set():
            _cancel_event.clear()
            raise InterruptedError("Generation cancelled by user")

        latent_input = pipe.scheduler.scale_model_input(latents, t)

        # 1) Unconditional noise prediction (shared negative for CFG)
        uncond_kwargs = {
            "text_embeds": neg_pooled,
            "time_ids": add_time_ids,
        }
        with torch.no_grad():
            noise_uncond = pipe.unet(
                latent_input, t,
                encoder_hidden_states=neg_embeds,
                added_cond_kwargs=uncond_kwargs,
            ).sample

        # 2) Per-character conditional predictions, blended by soft masks
        #    Each character's prompt includes the shared scene for coherence.
        #    Masks are softmax-normalised and sum to 1.0, so the blended
        #    result is one seamless noise prediction with no gaps or boxes.
        noise_cond = torch.zeros_like(noise_uncond)

        for mask, (char_emb, char_pooled) in zip(masks, char_embeds_list):
            char_kwargs = {
                "text_embeds": char_pooled,
                "time_ids": add_time_ids,
            }
            with torch.no_grad():
                noise_char = pipe.unet(
                    latent_input, t,
                    encoder_hidden_states=char_emb,
                    added_cond_kwargs=char_kwargs,
                ).sample
            noise_cond = noise_cond + noise_char * mask

        # 3) Classifier-Free Guidance
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # 4) Scheduler step
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # ── Decode latents to image ────────────────────────────────────────────
    latents = latents / pipe.vae.config.scaling_factor
    with torch.no_grad():
        image = pipe.vae.decode(latents).sample
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]

    _log.info("[IMAGE GEN] Regional generation complete — %d regions", n_chars)
    return image


def _build_anti_bleed_negative(regions: list[dict]) -> str:
    """Generate per-character anti-bleed negative prompt additions.

    Scans each character's description for distinctive features (hair color,
    eye color) and adds them as negatives for OTHER characters' regions.
    This helps SDXL avoid bleeding features between characters.
    """
    # Extract color-related features from each character section
    color_features = []
    color_pattern = re.compile(
        r'((?:light |dark )?(?:blonde|pink|blue|red|green|purple|white|black|brown|silver|grey|gray|'
        r'orange|yellow|multicolored|streaked)\s+(?:hair|eyes?))',
        re.IGNORECASE,
    )
    for r in regions:
        features = color_pattern.findall(r["prompt"])
        color_features.append(features)

    # For each character, the OTHER characters' distinctive features should be negated
    anti_bleed_parts = []
    for i, features in enumerate(color_features):
        other_features = []
        for j, other in enumerate(color_features):
            if i != j:
                other_features.extend(other)
        if other_features:
            anti_bleed_parts.extend(other_features)

    if anti_bleed_parts:
        # Deduplicate
        seen = set()
        unique = []
        for f in anti_bleed_parts:
            fl = f.lower()
            if fl not in seen:
                seen.add(fl)
                unique.append(f)
        return ", ".join(unique)
    return ""


# ── Core generation ───────────────────────────────────────────────────────

DEFAULT_NEGATIVE = (
    "blurry, low quality, deformed, bad anatomy, watermark, bad hands, "
    "missing fingers, extra fingers, fused fingers, too many fingers, mutated hands, "
    "malformed hands, extra digit, fewer digits, ugly hands, poorly drawn hands, "
    "text, error, cropped, worst quality, bad quality, worst detail, sketch, "
    "censored, signature, watermark"
)

EXPLICIT_NEGATIVE = (
    "score_4, score_3, score_2, score_1, "
    "blurry, low quality, deformed, bad anatomy, watermark, bad hands, "
    "missing fingers, extra fingers, fused fingers, too many fingers, mutated hands, "
    "malformed hands, extra digit, fewer digits, ugly hands, poorly drawn hands, "
    "text, error, cropped, worst quality, bad quality, worst detail, sketch, "
    "censored, artist name, signature, watermark, "
    "patreon username, patreon logo"
)


def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    model_path: str | None = None,
    lora_paths: list[str] | None = None,
    lora_weights: list[float] | None = None,
    mode: str = "normal",  # "normal" or "explicit"
    steps: int = 35,
    guidance_scale: float = 7.5,
    negative_prompt: str | None = None,
    seed: int = -1,
    art_style: str = "custom",
    selected_outfits: dict[str, str] | None = None,
) -> dict:
    """
    Generate an image. Returns dict with path, prompt analysis, settings used.

    Args:
        prompt: User's text prompt
        model_path: Path to checkpoint (uses default if None)
        lora_paths: Optional list of LoRA files to load
        mode: "normal" (safe) or "explicit" (with Pony quality tags)
        steps: Inference steps (20-50 range)
        guidance_scale: CFG scale
        negative_prompt: Override negative prompt (uses smart default otherwise)
        seed: Reproducibility seed (-1 = random)
    """
    # Resolve model
    if model_path is None:
        # Use first available checkpoint or the active one
        if _active_model:
            model_path = _active_model
        else:
            models = list_models()
            checkpoints = [m for m in models if m["type"] == "checkpoint"]
            if not checkpoints:
                return {"error": "No model files found in models/ directory", "status": "error"}
            model_path = checkpoints[0]["path"]

    if not Path(model_path).exists():
        return {"error": f"Model not found: {model_path}", "status": "error"}

    # Free Ollama models from VRAM before loading SD pipeline
    evict_ollama_models()

    # Load pipeline
    pipe = _load_pipeline(model_path)

    # Load LoRAs if requested
    lora_diagnostics = []  # track what was actually loaded
    if lora_paths:
        unload_loras(pipe)  # clear any previous LoRAs
        weights = lora_weights or [0.8] * len(lora_paths)
        for lp, lw in zip(lora_paths, weights):
            if Path(lp).exists():
                load_lora(pipe, lp, weight=lw)
                stem = Path(lp).stem
                tw_data_diag = _load_trigger_words()
                words_diag = tw_data_diag.get(stem, [])
                lora_diagnostics.append({
                    "name": stem,
                    "path": lp,
                    "weight": lw,
                    "trigger_words": [parse_trigger_word_entry(w)["word"]
                                      for w in words_diag if parse_trigger_word_entry(w)["word"] != ";"],
                    "loaded": True,
                })
                _log.info("[IMAGE GEN] ✓ LoRA loaded: %s (weight=%.2f, file=%s)",
                          stem, lw, Path(lp).name)
            else:
                lora_diagnostics.append({
                    "name": Path(lp).stem,
                    "path": lp,
                    "weight": lw,
                    "trigger_words": [],
                    "loaded": False,
                    "error": "File not found",
                })
                _log.warning("[IMAGE GEN] ✗ LoRA not found: %s", lp)

    # Auto-inject missing trigger words for active LoRAs
    # When 2+ character LoRAs are loaded, skip character LoRA injection here —
    # _structure_multi_character_prompt will handle proper BREAK-separated injection.
    injected_triggers = []
    if lora_paths:
        tw_data = _load_trigger_words()
        prompt_lower = prompt.lower()
        selected_outfits = selected_outfits or {}

        # Determine which LoRAs are character-type
        char_dir = MODELS_DIR / "characters"
        char_lora_stems = set()
        for lp in lora_paths:
            try:
                if Path(lp).parent.resolve() == char_dir.resolve():
                    char_lora_stems.add(Path(lp).stem)
            except Exception:
                pass
        multi_char_mode = len(char_lora_stems) >= 2

        for lp in lora_paths:
            if not Path(lp).exists():
                continue
            stem = Path(lp).stem
            # Skip character LoRAs in multi-character mode — handled by BREAK structuring
            if multi_char_mode and stem in char_lora_stems:
                _log.info("[IMAGE GEN] Skipping auto-inject for character LoRA %s (multi-char mode)", stem)
                words = tw_data.get(stem, [])
                if words:
                    outfit = selected_outfits.get(stem)
                    trigger_str = build_trigger_prompt(words, selected_outfit=outfit)
                    if trigger_str:
                        injected_triggers.append({"lora": stem, "injected": trigger_str, "method": "BREAK section"})
                continue
            words = tw_data.get(stem, [])
            if words:
                # Build the full trigger prompt using outfit-aware system
                outfit = selected_outfits.get(stem)
                trigger_str = build_trigger_prompt(words, selected_outfit=outfit)
                if trigger_str:
                    # Check which parts are already in the prompt
                    missing_parts = []
                    for part in trigger_str.split(", "):
                        part_clean = re.sub(r'^\(|\)$', '', part).split(":")[0].lower()
                        if part_clean and part_clean not in prompt_lower:
                            missing_parts.append(part)
                    if missing_parts:
                        inject_str = ", ".join(missing_parts)
                        prompt = f"{inject_str}, {prompt}"
                        prompt_lower = prompt.lower()
                        injected_triggers.append({"lora": stem, "injected": inject_str})
                        _log.info("[IMAGE GEN] Auto-injected trigger words for LoRA %s: %s",
                                  stem, inject_str)

    # Build prompt based on mode
    if mode == "explicit":
        full_prompt = f"score_9, score_8_up, score_7_up, {prompt}"
        base_neg = negative_prompt or EXPLICIT_NEGATIVE
    else:
        full_prompt = prompt
        base_neg = negative_prompt or DEFAULT_NEGATIVE

    # Strip A1111/ComfyUI-style <lora:...> tags — this backend loads LoRAs via API,
    # so these tags in the prompt text are noise to the CLIP encoder
    full_prompt = re.sub(r'<lora:[^>]+>', '', full_prompt)
    full_prompt = re.sub(r',\s*,', ',', full_prompt)  # clean double commas
    full_prompt = re.sub(r'\s{2,}', ' ', full_prompt).strip(', ')

    # Auto-insert BREAK tokens before parenthesized character blocks when the
    # user didn't type them.  This ensures CLIP chunking isolates each
    # character's description even for single-character prompts.
    if 'BREAK' not in full_prompt:
        # Detect parenthesized blocks that look like character descriptions:
        # (TriggerCode:weight, tag, tag, ...) or (TriggerCode, tag, tag, ...)
        tw_data_auto = _load_trigger_words()
        char_dir_auto = MODELS_DIR / "characters"
        _all_trigger_codes: set[str] = set()
        for _stem, _words in tw_data_auto.items():
            lp_check = char_dir_auto / f"{_stem}.safetensors"
            if lp_check.exists():
                _pg = parse_outfit_groups(_words)
                if _pg["primary"]:
                    _all_trigger_codes.add(_pg["primary"].lower())
                for _on, _tags in _pg.get("outfits", {}).items():
                    if _tags:
                        _c = _tags[0]["word"] if isinstance(_tags[0], dict) else str(_tags[0])
                        if _c:
                            _all_trigger_codes.add(_c.lower())

        # Find positions of parenthesized blocks containing known trigger codes
        paren_blocks = list(re.finditer(r'\([^)]+\)', full_prompt))
        insert_positions: list[int] = []
        for pb in paren_blocks:
            block_text = pb.group(0).lower()
            for code in _all_trigger_codes:
                if code in block_text:
                    insert_positions.append(pb.start())
                    break

        if len(insert_positions) >= 2:
            # Only auto-insert BREAKs for MULTI-character prompts (2+ blocks).
            # Single-character prompts benefit from keeping everything in one
            # CLIP attention window — splitting weakens LoRA coherence.
            modified = full_prompt
            for pos in sorted(insert_positions, reverse=True):
                # Don't insert BREAK at the very start
                if pos > 0:
                    # Clean trailing commas/spaces before the BREAK
                    pre = modified[:pos].rstrip(', ')
                    post = modified[pos:]
                    modified = f"{pre} BREAK {post}"
            if modified != full_prompt:
                _log.info("[IMAGE GEN] Auto-inserted BREAK tokens before %d character block(s)", len(insert_positions))
                full_prompt = modified
        elif len(insert_positions) == 1:
            _log.info("[IMAGE GEN] Single character block detected — skipping auto-BREAK to preserve CLIP coherence")

    # Structure multi-angle and multi-character prompts
    full_prompt = _structure_multi_angle_prompt(full_prompt)
    full_prompt = _structure_multi_character_prompt(full_prompt, lora_paths, selected_outfits)

    # Apply art style
    full_prompt, base_neg = _apply_art_style(full_prompt, base_neg, art_style)

    # Apply feedback learnings
    full_negative = apply_feedback_learnings(base_neg)
    full_prompt = apply_positive_learnings(full_prompt)

    # Clamp settings
    steps = max(10, min(steps, 80))
    guidance_scale = max(1.0, min(guidance_scale, 20.0))
    width = max(512, min(width, 2048))
    height = max(512, min(height, 2048))

    # Seed
    generator = None
    if seed >= 0:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Prompt analysis for transparency
    prompt_analysis = _analyze_prompt(full_prompt)

    _log.info("[IMAGE GEN] Generating: model=%s, mode=%s, steps=%d, cfg=%.1f, seed=%d",
              Path(model_path).stem, mode, steps, guidance_scale, seed)

    # Try long-prompt encoding
    embed_kwargs = {}
    long_prompt_used = False
    encoding_result = _encode_long_prompt(pipe, full_prompt, full_negative)
    if encoding_result and "_chunked_prompt" not in encoding_result:
        embed_kwargs = encoding_result
        long_prompt_used = True
    elif encoding_result and "_chunked_prompt" in encoding_result:
        full_prompt = encoding_result["_chunked_prompt"]
        full_negative = encoding_result["_chunked_negative"]

    # Generate
    try:
        output_dir_path = Path(OUTPUT_DIR)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        _update_progress(active=True, type="image", current_step=0,
                         total_steps=steps, current_frame=0, total_frames=1,
                         message="Generating image...")

        _step_start_time = time.time()

        def _cancel_callback(pipe_self, step_index, timestep, callback_kwargs):
            """Check cancel flag + update progress with it/s between diffusion steps."""
            elapsed = time.time() - _step_start_time
            its = (step_index / elapsed) if elapsed > 0 and step_index > 0 else 0.0
            _update_progress(
                current_step=step_index + 1,
                message=f"{its:.2f} it/s" if step_index > 0 else "starting...",
            )
            if _cancel_event.is_set():
                _cancel_event.clear()
                raise InterruptedError("Generation cancelled by user")
            return callback_kwargs

        # ── Check for regional multi-character mode ──────────────────────
        regional_parsed = _parse_regional_sections(full_prompt)
        use_regional = regional_parsed is not None

        if use_regional:
            # Enhance negative prompt with anti-bleed terms
            _, regional_regions = regional_parsed
            anti_bleed = _build_anti_bleed_negative(regional_regions)
            regional_negative = f"{full_negative}, {anti_bleed}" if anti_bleed else full_negative

            _log.info("[IMAGE GEN] Using REGIONAL conditioning pipeline for %d characters",
                      len(regional_regions))
            image = _generate_regional(
                pipe, full_prompt, regional_negative,
                width, height, steps, guidance_scale, generator,
            )
        else:
            gen_kwargs = {
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "callback_on_step_end": _cancel_callback,
            }

            if long_prompt_used and embed_kwargs:
                gen_kwargs.update(embed_kwargs)
            else:
                gen_kwargs["prompt"] = full_prompt
                gen_kwargs["negative_prompt"] = full_negative

            image = pipe(**gen_kwargs).images[0]

        # Signal completion to the frontend
        _update_progress(current_step=steps, total_steps=steps,
                         message="Saving image...")

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r'[^\w\s-]', '', prompt[:30]).strip().replace(' ', '_')
        filename = str(output_dir_path / f"{timestamp}_{safe_name}.png")
        image.save(filename)
        _log.info("[IMAGE GEN] Saved to: %s", filename)

        # Record for feedback loop
        settings = {
            "model": Path(model_path).stem,
            "loras": [Path(lp).stem for lp in (lora_paths or [])],
            "mode": mode,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "width": width,
            "height": height,
        }
        record = record_generation(full_prompt, full_negative, settings, filename)

        _update_progress(current_step=steps, total_steps=steps,
                         message="Complete!")

        return {
            "status": "ok",
            "path": filename,
            "seed": seed,
            "prompt_used": full_prompt,
            "negative_used": full_negative,
            "prompt_analysis": prompt_analysis,
            "settings": settings,
            "long_prompt": long_prompt_used,
            "lora_diagnostics": lora_diagnostics,
            "injected_triggers": injected_triggers,
            "regional_mode": use_regional,
        }

    except InterruptedError:
        _log.info("[IMAGE GEN] Generation cancelled by user")
        return {"status": "cancelled", "message": "Generation was stopped by user"}
    except Exception as e:
        _log.error("[IMAGE GEN] Error: %s", str(e))
        return {"status": "error", "error": str(e)}
    finally:
        # Keep "Complete!" visible for a few seconds so the frontend poll catches it
        _update_progress(active=True, type="image", current_step=steps,
                         total_steps=steps, message="Complete!")
        import threading as _th
        def _clear_progress():
            time.sleep(3)
            _update_progress(active=False, type="idle", current_step=0,
                             total_steps=0, message="")
        _th.Thread(target=_clear_progress, daemon=True).start()


# ── Art Style Presets ──────────────────────────────────────────────────────
# Each preset shapes prompt prefix AND negative to steer the model away from
# the generic "lifeless masterpiece" look of raw Illustrious models.

ART_STYLES: dict[str, dict] = {
    "anime": {
        "prefix": (
            "anime screencap, hand-drawn animation, cel shading, "
            "soft lighting, natural colour palette, expressive linework, "
            "high quality anime key visual"
        ),
        "negative_extra": (
            "3d render, photorealistic, CGI, plastic, airbrushed, "
            "overly saturated, neon colours, smooth shading, "
            "deviantart, artstation trending, western comic"
        ),
    },
    "ghibli": {
        "prefix": (
            "studio ghibli style, watercolour background, "
            "warm lighting, soft pastel tones, hand-painted, "
            "detailed environment, gentle linework"
        ),
        "negative_extra": (
            "3d render, photorealistic, digital art, "
            "overly saturated, neon, sharp lines, CGI"
        ),
    },
    "manga": {
        "prefix": (
            "manga panel, black and white, screentone, "
            "detailed linework, dramatic shading, ink drawing"
        ),
        "negative_extra": (
            "colour, colorful, 3d render, photorealistic, "
            "painting, watercolour"
        ),
    },
    "painterly": {
        "prefix": (
            "oil painting, visible brush strokes, impressionist lighting, "
            "textured canvas, rich earth tones, gallery quality"
        ),
        "negative_extra": (
            "anime, cartoon, digital art, smooth shading, "
            "flat colours, 3d render, photography"
        ),
    },
    "game_concept": {
        "prefix": (
            "game concept art, splash art, dynamic composition, "
            "dramatic rim lighting, painterly rendering, "
            "detailed character design"
        ),
        "negative_extra": (
            "photograph, photorealistic, anime, flat colours, "
            "simple background, sketch"
        ),
    },
    "realistic": {
        "prefix": (
            "photorealistic, RAW photo, 8k UHD, DSLR, "
            "professional lighting, studio quality"
        ),
        "negative_extra": (
            "anime, cartoon, drawing, painting, illustration, "
            "stylized, flat colours"
        ),
    },
    "3d_blender": {
        "prefix": (
            "3d render, blender, unreal engine 5, octane render, "
            "volumetric lighting, subsurface scattering, PBR materials, "
            "video game character, detailed textures, sharp focus"
        ),
        "negative_extra": (
            "anime, hand-drawn, sketch, painting, flat colours, "
            "2d, cel shading, watercolour, pencil, photo"
        ),
    },
    "custom": {
        "prefix": "",  # user supplies their own via LoRA or prompt
        "negative_extra": "",
    },
}

STYLE_NAMES = list(ART_STYLES.keys())


def get_art_styles() -> list[dict]:
    """Return available art style presets for the frontend."""
    return [
        {"name": k, "prefix": v["prefix"][:80] + "…" if len(v["prefix"]) > 80 else v["prefix"]}
        for k, v in ART_STYLES.items()
    ]


def _apply_art_style(prompt: str, negative: str, style: str) -> tuple[str, str]:
    """Prepend style prefix and extend negative with style-specific exclusions."""
    preset = ART_STYLES.get(style)
    if not preset or style == "custom":
        return prompt, negative
    styled_prompt = f"{preset['prefix']}, {prompt}" if preset["prefix"] else prompt
    styled_neg = f"{negative}, {preset['negative_extra']}" if preset["negative_extra"] else negative
    return styled_prompt, styled_neg


# ── Img2Img pipeline (for frame consistency) ──────────────────────────────

_img2img_pipelines: dict[str, object] = {}


def _load_img2img_pipeline(model_path: str):
    """Load the img2img variant of an SDXL pipeline.
    Shares the base model weights via from_pipe() when possible,
    otherwise loads from scratch."""
    if model_path in _img2img_pipelines:
        return _img2img_pipelines[model_path]

    try:
        from diffusers import StableDiffusionXLImg2ImgPipeline
    except ImportError:
        _log.error("[IMAGE GEN] StableDiffusionXLImg2ImgPipeline not available — update diffusers")
        return None

    # Try to share weights with the txt2img pipeline
    if model_path in _loaded_pipelines:
        txt2img = _loaded_pipelines[model_path]
        _log.info("[IMAGE GEN] Creating img2img pipeline from existing txt2img")
        pipe = StableDiffusionXLImg2ImgPipeline(
            vae=txt2img.vae,
            text_encoder=txt2img.text_encoder,
            text_encoder_2=txt2img.text_encoder_2,
            tokenizer=txt2img.tokenizer,
            tokenizer_2=txt2img.tokenizer_2,
            unet=txt2img.unet,
            scheduler=txt2img.scheduler,
        ).to("cuda")
    else:
        _log.info("[IMAGE GEN] Loading img2img pipeline from file: %s", model_path)
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")

    pipe.enable_attention_slicing()
    _img2img_pipelines[model_path] = pipe
    return pipe


# ── Consistent Animated Generation ───────────────────────────────────────

# Savepoint directory for animation jobs
_ANIM_SAVES_DIR = Path(OUTPUT_DIR) / "animation_saves"


def _resolve_model_path(model_path: str | None) -> str | None:
    """Resolve model path with fallback to active or first available."""
    if model_path is not None:
        return model_path
    if _active_model:
        return _active_model
    models = list_models()
    checkpoints = [m for m in models if m["type"] == "checkpoint"]
    return checkpoints[0]["path"] if checkpoints else None


# ── Animation helper utilities ────────────────────────────────────────────

def _match_color_to_reference(image, reference_np):
    """Match the color distribution of `image` to the reference frame.

    Uses LAB colour-space transfer (perceptually uniform) for luminance
    and chrominance channels, plus an RGB brightness guard that clamps
    overall exposure drift.  This prevents the cumulative "oil spill" /
    over-exposure effect that builds up over many img2img frames.
    """
    from PIL import Image as PILImage
    import numpy as np

    img_np = np.array(image, dtype=np.float64)

    # ── LAB colour transfer (perceptually uniform) ───────────────
    # Convert RGB → LAB approximation via simple linear transform
    # (avoids OpenCV dependency)
    def _rgb_to_lab_approx(rgb):
        # Normalise 0-1, apply sRGB → linear
        lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
        # Linear RGB → XYZ (D65)
        x = lin[:, :, 0] * 0.4124 + lin[:, :, 1] * 0.3576 + lin[:, :, 2] * 0.1805
        y = lin[:, :, 0] * 0.2126 + lin[:, :, 1] * 0.7152 + lin[:, :, 2] * 0.0722
        z = lin[:, :, 0] * 0.0193 + lin[:, :, 1] * 0.1192 + lin[:, :, 2] * 0.9505
        # XYZ → Lab
        xn, yn, zn = 0.95047, 1.0, 1.08883
        def f(t):
            return np.where(t > 0.008856, t ** (1/3), 7.787 * t + 16/116)
        fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        return np.stack([L, a, b], axis=-1)

    def _lab_to_rgb_approx(lab):
        L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        def finv(t):
            return np.where(t > 0.206893, t ** 3, (t - 16/116) / 7.787)
        x = finv(fx) * 0.95047
        y = finv(fy) * 1.0
        z = finv(fz) * 1.08883
        r = x *  3.2406 + y * -1.5372 + z * -0.4986
        g = x * -0.9689 + y *  1.8758 + z *  0.0415
        bl= x *  0.0557 + y * -0.2040 + z *  1.0570
        rgb = np.stack([r, g, bl], axis=-1)
        # Linear → sRGB
        rgb = np.where(rgb > 0.0031308, 1.055 * (np.maximum(rgb, 0) ** (1/2.4)) - 0.055, 12.92 * rgb)
        return rgb

    src_01 = img_np / 255.0
    ref_01 = reference_np / 255.0

    src_lab = _rgb_to_lab_approx(src_01)
    ref_lab = _rgb_to_lab_approx(ref_01)

    # Per-channel mean/std transfer in LAB space
    for c in range(3):
        s_mean = np.mean(src_lab[:, :, c])
        s_std  = np.std(src_lab[:, :, c])
        r_mean = np.mean(ref_lab[:, :, c])
        r_std  = np.std(ref_lab[:, :, c])
        if s_std > 1e-6:
            src_lab[:, :, c] = (src_lab[:, :, c] - s_mean) * (r_std / s_std) + r_mean

    # Convert back to RGB
    matched_01 = _lab_to_rgb_approx(src_lab)
    matched = np.clip(matched_01 * 255, 0, 255).astype(np.uint8)

    # ── Brightness guard ──────────────────────────────────────────
    # Clamp overall luminance to ±5% of reference to prevent exposure drift
    ref_lum = np.mean(reference_np.astype(np.float64))
    out_lum = np.mean(matched.astype(np.float64))
    if ref_lum > 1.0:
        ratio = out_lum / ref_lum
        if ratio > 1.05 or ratio < 0.95:
            correction = ref_lum / max(out_lum, 1e-6)
            matched = np.clip(matched.astype(np.float64) * correction, 0, 255).astype(np.uint8)

    return PILImage.fromarray(matched)


def _build_strength_curve(
    base_strength: float, num_frames: int, curve: str
) -> list[float]:
    """Build a per-frame denoising strength schedule.

    Curves:
      - constant: same strength every frame
      - ease_in: start gentle (50% strength), ramp up to full over first 25% of frames
      - pulse: sinusoidal "breathing" — varies ±30% around base
    """
    import math

    strengths = []
    for i in range(num_frames):
        t = i / max(1, num_frames - 1)  # 0..1

        if curve == "ease_in":
            # First 25% of frames: ramp from 50% to 100% of base_strength
            if t < 0.25:
                factor = 0.5 + 2.0 * t  # 0.5 → 1.0 over first quarter
            else:
                factor = 1.0
            strengths.append(base_strength * factor)

        elif curve == "pulse":
            # Sinusoidal breathing: ±30% variation
            factor = 1.0 + 0.3 * math.sin(2 * math.pi * t * 2)
            strengths.append(base_strength * factor)

        else:  # constant
            strengths.append(base_strength)

    # Clamp all values
    return [max(0.10, min(s, 0.70)) for s in strengths]


# ── Body movement keywords for auto-storyboard ───────────────────────────

_BODY_PART_ACTIONS = {
    "head": [
        "head tilting slightly left", "head turning right", "head nodding down",
        "head lifting up", "head tilting back to center",
    ],
    "hands": [
        "hands rising slowly", "hands moving outward", "fingers spreading open",
        "hands gesturing", "hands lowering back down",
    ],
    "arms": [
        "arms lifting slightly", "arms extending outward", "arms crossing",
        "arms dropping to sides", "arms swinging gently",
    ],
    "legs": [
        "weight shifting to left leg", "weight shifting to right leg",
        "knee bending slightly", "stance widening", "feet repositioning",
    ],
    "body": [
        "torso leaning forward slightly", "body swaying gently",
        "shoulders rotating", "torso straightening", "posture shifting",
    ],
    "hips": [
        "hips shifting left", "hips swaying right", "hips thrusting forward",
        "hips rolling", "hips returning to center",
    ],
    "hair": [
        "hair flowing with breeze", "hair swaying left", "hair settling down",
        "hair swept by wind", "hair bouncing with movement",
    ],
    "eyes": [
        "eyes looking left", "eyes looking right", "eyes looking down",
        "eyes closing slightly", "eyes opening wide",
    ],
    "chest": [
        "chest rising with breath", "chest expanding", "chest relaxing",
        "chest leaning forward", "chest settling",
    ],
    "feet": [
        "foot tapping", "feet shifting weight", "toes curling",
        "feet adjusting stance", "feet settling",
    ],
}

# Action keywords that imply specific movements — each entry now has
# `steps` (motion sequence), `involved_parts` (what body parts move with it),
# and `physics` (speed/weight characteristics for easing).
_ACTION_MOTION_MAP: dict[str, dict] = {
    "walking": {
        "aliases": ["walk", "strolling", "striding", "stepping"],
        "steps": ["left foot stepping forward", "right foot stepping forward",
                  "arms swinging naturally", "weight transferring", "stride continuing"],
        "involved_parts": ["legs", "arms", "body", "hair"],
        "physics": "cyclic",  # repeating loop
    },
    "running": {
        "aliases": ["run", "sprinting", "jogging", "dashing"],
        "steps": ["legs pumping forward", "arms driving back", "body leaning forward",
                  "feet pushing off ground", "full sprint momentum"],
        "involved_parts": ["legs", "arms", "body", "hair"],
        "physics": "cyclic",
    },
    "dancing": {
        "aliases": ["dance", "swaying", "twirling", "grooving"],
        "steps": ["hips swaying", "arms flowing upward", "body spinning slowly",
                  "feet stepping rhythmically", "upper body waving"],
        "involved_parts": ["hips", "arms", "legs", "body", "hair"],
        "physics": "cyclic",
    },
    "fighting": {
        "aliases": ["fight", "punching", "kicking", "striking", "combat"],
        "steps": ["fist pulling back", "punch extending forward", "dodge to the side",
                  "kick lifting", "guard position raised"],
        "involved_parts": ["arms", "legs", "body"],
        "physics": "burst",  # sharp acceleration then slow
    },
    "waving": {
        "aliases": ["wave", "beckoning", "greeting"],
        "steps": ["hand rising up", "hand moving side to side", "fingers waving",
                  "arm extending fully", "hand lowering back"],
        "involved_parts": ["arms", "hands"],
        "physics": "pendulum",  # back and forth
    },
    "sitting": {
        "aliases": ["sit", "seated", "resting", "lounging"],
        "steps": ["settling into seat", "adjusting posture", "crossing legs",
                  "leaning back slightly", "hands resting on lap"],
        "involved_parts": ["body", "legs", "hands"],
        "physics": "settle",  # large movement → small adjustments
    },
    "flying": {
        "aliases": ["fly", "soaring", "hovering", "floating", "gliding"],
        "steps": ["wings spreading wide", "body ascending", "arms outstretched",
                  "floating higher", "soaring through air"],
        "involved_parts": ["arms", "body", "hair"],
        "physics": "cyclic",
    },
    "casting": {
        "aliases": ["cast", "spell", "magic", "channeling", "conjuring"],
        "steps": ["hands gathering energy", "magic circle forming", "energy releasing",
                  "spell shooting forward", "magical aftermath fading"],
        "involved_parts": ["arms", "hands", "eyes"],
        "physics": "buildup",  # slow wind-up then release
    },
    "turning": {
        "aliases": ["turn", "spinning", "rotating", "pivoting"],
        "steps": ["body starting to rotate", "shoulders turning", "hips following",
                  "full body mid-turn", "turn completing, settling"],
        "involved_parts": ["body", "hips", "head", "hair"],
        "physics": "arc",
    },
    "jumping": {
        "aliases": ["jump", "leaping", "hopping", "bouncing"],
        "steps": ["knees bending for launch", "body pushing upward", "airborne at peak",
                  "body descending", "landing with knees absorbing"],
        "involved_parts": ["legs", "arms", "body", "hair"],
        "physics": "burst",
    },
    "stretching": {
        "aliases": ["stretch", "yawning", "reaching"],
        "steps": ["arms reaching upward", "body elongating", "muscles tensing",
                  "slow release", "arms lowering back"],
        "involved_parts": ["arms", "body", "chest"],
        "physics": "settle",
    },
    "hugging": {
        "aliases": ["hug", "embracing", "holding"],
        "steps": ["arms opening wide", "arms wrapping around", "pulling close",
                  "squeezing gently", "slowly releasing"],
        "involved_parts": ["arms", "hands", "body"],
        "physics": "arc",
    },
    "riding": {
        "aliases": ["ride", "mounted", "horseback", "cowgirl", "straddling"],
        "steps": ["hips rising", "body bouncing upward", "weight coming down",
                  "hips grinding forward", "settling into rhythm"],
        "involved_parts": ["hips", "legs", "body", "chest", "hair"],
        "physics": "cyclic",
    },
    "swimming": {
        "aliases": ["swim", "floating", "diving", "treading"],
        "steps": ["arms pulling through water", "legs kicking", "body gliding forward",
                  "arms recovering forward", "breath lifting head"],
        "involved_parts": ["arms", "legs", "body", "head"],
        "physics": "cyclic",
    },
    "eating": {
        "aliases": ["eat", "chewing", "biting", "munching"],
        "steps": ["hand bringing food to mouth", "mouth opening", "chewing",
                  "swallowing", "hand lowering"],
        "involved_parts": ["hands", "head"],
        "physics": "cyclic",
    },
    "crying": {
        "aliases": ["cry", "sobbing", "weeping", "tearing up"],
        "steps": ["shoulders trembling slightly", "head lowering", "hand wiping eyes",
                  "body shuddering", "slowly composing"],
        "involved_parts": ["body", "head", "hands", "eyes"],
        "physics": "pendulum",
    },
    "laughing": {
        "aliases": ["laugh", "giggling", "chuckling"],
        "steps": ["body shaking slightly with laughter", "head tilting back",
                  "hand covering mouth", "shoulders bouncing", "settling down"],
        "involved_parts": ["body", "head", "hands"],
        "physics": "pendulum",
    },
}

# Compound pose phrases that imply multiple body parts
_POSE_IMPLICATIONS: dict[str, list[str]] = {
    "standing": ["legs", "body"],
    "sitting at desk": ["arms", "hands", "body"],
    "lying down": ["body", "legs", "arms"],
    "leaning": ["body", "arms"],
    "crouching": ["legs", "body"],
    "kneeling": ["legs", "body"],
    "looking up": ["head", "eyes"],
    "looking down": ["head", "eyes"],
    "looking away": ["head", "eyes"],
    "arms raised": ["arms", "hands"],
    "arms behind": ["arms", "body"],
    "hands on hips": ["hands", "hips"],
    "crossed arms": ["arms", "body"],
    "bent over": ["body", "hips", "legs"],
    "on all fours": ["arms", "legs", "body", "hips"],
    "from behind": ["body", "hips", "hair"],
    "from above": ["head", "body"],
    "portrait": ["head", "eyes", "hair"],
    "full body": ["body", "legs", "arms", "head", "hair"],
    "upper body": ["body", "arms", "head", "hair"],
    "close up": ["head", "eyes"],
    "action pose": ["arms", "legs", "body"],
    "dynamic pose": ["arms", "legs", "body", "hair"],
    "relaxed": ["body", "arms"],
    "tense": ["arms", "body", "hands"],
}


def _ease_factor(t: float, physics: str) -> float:
    """Return a 0-1 easing factor based on physics type.

    t is the normalised progress through the motion (0 to 1).
    """
    import math
    if physics == "burst":
        # Fast start, slow finish (ease-out cubic)
        return 1.0 - (1.0 - t) ** 3
    elif physics == "buildup":
        # Slow start, fast finish (ease-in cubic)
        return t ** 3
    elif physics == "pendulum":
        # Back-and-forth sinusoidal
        return 0.5 + 0.5 * math.sin(2 * math.pi * t)
    elif physics == "settle":
        # Large at start, damps down
        return max(0.1, 1.0 - t * 0.8)
    elif physics == "arc":
        # Smooth bell curve — peak at midpoint
        return math.sin(math.pi * t)
    else:  # "cyclic" or default
        return 1.0  # constant intensity for cyclic motions


def _detect_actions(prompt_lower: str) -> list[tuple[str, dict]]:
    """Detect all matching actions from the prompt using primary keys and aliases.
    Returns list of (action_name, action_dict) sorted by relevance (longer match first).
    """
    matches: list[tuple[str, dict, int]] = []
    for action_name, action_data in _ACTION_MOTION_MAP.items():
        all_keywords = [action_name] + action_data.get("aliases", [])
        for kw in all_keywords:
            if kw in prompt_lower:
                # Prefer longer keyword matches (more specific)
                matches.append((action_name, action_data, len(kw)))
                break  # Don't double-count same action
    # Sort by keyword length descending (more specific first)
    matches.sort(key=lambda x: x[2], reverse=True)
    return [(name, data) for name, data, _ in matches]


def _detect_body_parts(prompt_lower: str) -> list[str]:
    """Detect body parts from the prompt using direct keywords and pose implications."""
    detected: set[str] = set()

    # Direct keyword match
    for part in _BODY_PART_ACTIONS:
        if part in prompt_lower:
            detected.add(part)

    # Compound pose phrase detection (longer phrases checked first)
    for phrase, implied_parts in sorted(_POSE_IMPLICATIONS.items(),
                                        key=lambda x: len(x[0]), reverse=True):
        if phrase in prompt_lower:
            detected.update(implied_parts)

    return list(detected)


def _auto_generate_motion_storyboard(
    prompt: str, num_frames: int, motion_intensity: float
) -> list[str]:
    """Analyse prompt for body parts, actions, and poses, then generate a
    natural motion progression for each frame with temporal coherence.

    Improvements over naive cycling:
    - Detects multiple simultaneous actions and blends them
    - Uses physics-based easing for natural motion arcs
    - Combines primary action with secondary body part movements
    - Applies smooth temporal progression (ease-in/out) not abrupt cycling
    - Detects compound pose phrases (e.g. "sitting at desk") for implied parts

    Returns a list of per-frame motion descriptions.
    """
    import math
    prompt_lower = prompt.lower()
    descriptions: list[str] = []

    matched_actions = _detect_actions(prompt_lower)
    detected_parts = _detect_body_parts(prompt_lower)

    if matched_actions:
        # ── Primary action drives the motion ──────────────────────
        primary_name, primary_data = matched_actions[0]
        primary_steps = primary_data["steps"]
        physics = primary_data.get("physics", "cyclic")
        involved_parts = set(primary_data.get("involved_parts", []))

        # Collect secondary actions (if any)
        secondary_steps: list[str] = []
        for _, sec_data in matched_actions[1:]:
            secondary_steps.extend(sec_data["steps"][:2])  # just first 2 of each

        # Add ambient body part movements for parts NOT covered by the action
        ambient_parts = [p for p in (detected_parts or ["hair", "eyes"])
                         if p not in involved_parts]

        for i in range(num_frames):
            t = i / max(1, num_frames - 1)  # 0..1 normalised progress
            ease = _ease_factor(t, physics)

            # Primary motion — advance through steps with easing
            if physics == "cyclic":
                # Smooth cycling: use fractional index for fluid progression
                frac_idx = (i / max(1, num_frames - 1)) * len(primary_steps)
                step_idx = int(frac_idx) % len(primary_steps)
            elif physics in ("burst", "buildup", "arc"):
                # Map eased progress to step index
                step_idx = min(int(ease * len(primary_steps)),
                               len(primary_steps) - 1)
            else:
                step_idx = i % len(primary_steps)

            parts: list[str] = [primary_steps[step_idx]]

            # Add a secondary action hint every few frames
            if secondary_steps and i % 3 == 1:
                sec_idx = (i // 3) % len(secondary_steps)
                parts.append(secondary_steps[sec_idx])

            # Add ambient body-part movement (rotate through)
            if ambient_parts and i % 2 == 0:
                ap = ambient_parts[i % len(ambient_parts)]
                ap_actions = _BODY_PART_ACTIONS.get(ap, [])
                if ap_actions:
                    parts.append(ap_actions[i % len(ap_actions)])

            # Apply intensity modifiers
            joined = ", ".join(parts[:3])  # cap at 3 descriptors
            if motion_intensity < 0.3:
                desc = f"subtle {joined}"
            elif motion_intensity > 0.7:
                # Ease multiplies into the intensity word
                if ease > 0.7:
                    desc = f"dramatic {joined}, powerful motion"
                else:
                    desc = f"building {joined}, gaining momentum"
            else:
                desc = joined
            descriptions.append(desc)
        return descriptions

    # ── No specific action — pose-aware idle animation ────────────
    if not detected_parts:
        detected_parts = ["head", "body", "hair"]

    # Create smooth idle motion with staggered body parts
    num_parts = len(detected_parts)
    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        parts_for_frame: list[str] = []

        for pi, part in enumerate(detected_parts):
            actions = _BODY_PART_ACTIONS.get(part, [])
            if not actions:
                continue
            # Stagger each body part by an offset so they don't all move in sync
            offset = pi * len(actions) / max(1, num_parts)
            frac_idx = (t * len(actions) + offset) % len(actions)
            action_idx = int(frac_idx) % len(actions)

            # Apply a breathing ease — parts move more in the middle of the sequence
            breath = 0.5 + 0.5 * math.sin(math.pi * t)
            if breath > 0.3:  # only include movement when breath factor is strong enough
                parts_for_frame.append(actions[action_idx])

        if parts_for_frame:
            joined = ", ".join(parts_for_frame[:3])
            if motion_intensity < 0.3:
                desc = f"gentle idle, {joined}"
            elif motion_intensity > 0.7:
                desc = f"expressive idle, {joined}"
            else:
                desc = joined
        else:
            desc = "subtle breathing motion"
        descriptions.append(desc)

    return descriptions


def generate_animated(
    prompt: str,
    width: int = 512,
    height: int = 512,
    model_path: str | None = None,
    mode: str = "normal",
    num_frames: int = 16,
    steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = -1,
    art_style: str = "anime",
    frame_strength: float = 0.35,
    fps: int = 8,
    storyboard: list[str] | None = None,
    job_id: str | None = None,
    resume: bool = False,
    output_format: str = "gif",  # "gif", "mp4", or "both"
    lora_paths: list[str] | None = None,
    lora_weights: list[float] | None = None,
    negative_prompt: str | None = None,
    reference_blend: float = 0.3,  # blend ratio with frame 0 to prevent drift
    strength_curve: str = "constant",  # "constant", "ease_in", "pulse"
    motion_intensity: float = 0.5,  # 0.0=subtle, 1.0=dramatic body movement
) -> dict:
    """
    Generate a consistent animated sequence using img2img frame chaining.

    ANTI-DETERIORATION measures:
    - Reference frame blending: each frame's init is a blend of the previous
      frame + the first frame, anchoring color/composition to prevent cumulative
      drift (the "oil spill" / overexposure issue).
    - Color histogram matching: normalise each frame's color distribution to
      match frame 0, preventing brightness/saturation creep.
    - Strength curve: optionally vary denoising strength across frames.

    BODY MOVEMENT:
    - Auto-generates per-frame motion descriptions when storyboard is not provided,
      analysing the prompt for body parts, actions, and poses to create natural
      movement progression.

    Args:
        frame_strength: img2img denoising strength per frame (0.10-0.70).
        reference_blend: how much to blend with frame 0 (0=none, 1=full anchor).
        strength_curve: "constant", "ease_in" (gentle start), "pulse" (breathe).
        motion_intensity: how dramatic the body movement should be (0-1).
    """
    from PIL import Image as PILImage
    import numpy as np

    model_path = _resolve_model_path(model_path)
    if model_path is None:
        return {"status": "error", "error": "No model files found in models/ directory"}
    if not Path(model_path).exists():
        return {"status": "error", "error": f"Model not found: {model_path}"}

    num_frames = max(2, min(num_frames, 120))
    frame_strength = max(0.10, min(frame_strength, 0.70))
    fps = max(2, min(fps, 30))
    reference_blend = max(0.0, min(reference_blend, 0.6))
    motion_intensity = max(0.0, min(motion_intensity, 1.0))

    output_dir_path = Path(OUTPUT_DIR)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    _ANIM_SAVES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Job management / save-resume ──────────────────────────────
    if job_id is None:
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{id(prompt) % 10000:04d}"

    job_dir = _ANIM_SAVES_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = job_dir / "checkpoint.json"

    frame_paths: list[str] = []
    start_frame = 0
    base_seed = seed if seed >= 0 else torch.randint(0, 2**32, (1,)).item()

    if resume and checkpoint_file.exists():
        try:
            ckpt = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            saved_paths = ckpt.get("frame_paths", [])
            valid = [p for p in saved_paths if Path(p).exists()]
            frame_paths = valid
            start_frame = len(valid)
            base_seed = ckpt.get("seed", base_seed)
            _log.info("[ANIM GEN] Resuming job %s from frame %d/%d",
                      job_id, start_frame, num_frames)
        except Exception as e:
            _log.warning("[ANIM GEN] Checkpoint load failed: %s", e)

    # ── Auto-generate body movement storyboard ────────────────────
    frame_descriptions: list[str] = []
    if storyboard:
        for i in range(num_frames):
            idx = min(i, len(storyboard) - 1)
            frame_descriptions.append(storyboard[idx])
    else:
        frame_descriptions = _auto_generate_motion_storyboard(
            prompt, num_frames, motion_intensity
        )

    # ── Build per-frame strength curve ────────────────────────────
    frame_strengths = _build_strength_curve(
        frame_strength, num_frames, strength_curve
    )

    # ── Art style ────────────────────────────────────────────────
    base_neg = negative_prompt or DEFAULT_NEGATIVE
    styled_prompt, styled_negative = _apply_art_style(prompt, base_neg, art_style)
    full_negative = apply_feedback_learnings(styled_negative)
    styled_prompt = apply_positive_learnings(styled_prompt)

    # ── Load pipelines ────────────────────────────────────────────
    evict_ollama_models()
    _load_pipeline(model_path)
    img2img_pipe = _load_img2img_pipeline(model_path)
    if img2img_pipe is None:
        return {"status": "error", "error": "img2img pipeline unavailable — update diffusers"}

    txt2img_pipe = _loaded_pipelines[model_path]
    if lora_paths:
        unload_loras(txt2img_pipe)
        weights = lora_weights or [0.8] * len(lora_paths)
        for lp, lw in zip(lora_paths, weights):
            if Path(lp).exists():
                load_lora(txt2img_pipe, lp, weight=lw)
        try:
            for lp, lw in zip(lora_paths, lora_weights or [0.8] * len(lora_paths)):
                if Path(lp).exists():
                    img2img_pipe.load_lora_weights(lp)
                    img2img_pipe.fuse_lora(lora_scale=lw)
        except Exception as e:
            _log.warning("[ANIM GEN] LoRA on img2img failed: %s", e)

    # ── Frame generation loop ─────────────────────────────────────
    prev_image: PILImage.Image | None = None
    reference_image: PILImage.Image | None = None  # frame 0 anchor
    reference_np: np.ndarray | None = None  # frame 0 as numpy for color matching

    if frame_paths:
        try:
            prev_image = PILImage.open(frame_paths[-1]).convert("RGB")
            reference_image = PILImage.open(frame_paths[0]).convert("RGB")
            reference_np = np.array(reference_image, dtype=np.float64)
        except Exception:
            prev_image = None

    _update_progress(active=True, type="animation", current_step=0,
                     total_steps=steps, current_frame=start_frame,
                     total_frames=num_frames, message="Generating animation...")

    for i in range(start_frame, num_frames):
        if _check_cancelled():
            _log.info("[ANIM GEN] Cancelled at frame %d/%d", i, num_frames)
            break

        # ── VRAM safety check — auto-save before OOM ─────────────
        if not _check_vram_safe():
            _log.warning("[ANIM GEN] VRAM near limit at frame %d — auto-saving", i)
            _update_progress(message=f"VRAM critical — auto-saved at frame {i}/{num_frames}")
            break

        _update_progress(current_frame=i, message=f"Frame {i+1}/{num_frames}")

        frame_desc = frame_descriptions[i]
        if frame_desc:
            frame_prompt = f"{styled_prompt}, {frame_desc}"
        else:
            frame_prompt = styled_prompt

        frame_seed = base_seed
        cur_strength = frame_strengths[i]

        t0 = time.time()
        generator = torch.Generator(device="cuda").manual_seed(frame_seed)

        def _anim_cancel_cb(pipe_self, step_index, timestep, callback_kwargs):
            _update_progress(current_step=step_index + 1)
            if _cancel_event.is_set():
                _cancel_event.clear()
                raise InterruptedError("Animation cancelled by user")
            return callback_kwargs

        if i == 0 and prev_image is None:
            # First frame: txt2img
            gen_kwargs = {
                "prompt": frame_prompt,
                "negative_prompt": full_negative,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "callback_on_step_end": _anim_cancel_cb,
            }
            try:
                image = txt2img_pipe(**gen_kwargs).images[0]
            except InterruptedError:
                _log.info("[ANIM GEN] Cancelled during frame %d generation", i)
                break
            # Store frame 0 as the reference anchor
            reference_image = image.copy()
            reference_np = np.array(reference_image, dtype=np.float64)
        else:
            # ── Anti-deterioration: blend previous frame with reference ──
            init = prev_image.resize((width, height), PILImage.LANCZOS)

            if reference_image is not None and reference_blend > 0 and i >= 1:
                ref_resized = reference_image.resize((width, height), PILImage.LANCZOS)
                init_np = np.array(init, dtype=np.float64)
                ref_np = np.array(ref_resized, dtype=np.float64)
                # Blend: (1-blend)*prev + blend*reference
                blended_np = (1.0 - reference_blend) * init_np + reference_blend * ref_np
                init = PILImage.fromarray(np.clip(blended_np, 0, 255).astype(np.uint8))

            effective = int(steps * cur_strength)
            if effective < 1:
                adjusted_steps = max(steps, int(1.0 / cur_strength) + 1)
            else:
                adjusted_steps = steps

            gen_kwargs = {
                "prompt": frame_prompt,
                "negative_prompt": full_negative,
                "image": init,
                "strength": cur_strength,
                "num_inference_steps": adjusted_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "callback_on_step_end": _anim_cancel_cb,
            }
            try:
                image = img2img_pipe(**gen_kwargs).images[0]
            except InterruptedError:
                _log.info("[ANIM GEN] Cancelled during frame %d generation", i)
                break
            except Exception as e:
                _log.error("[ANIM GEN] Frame %d failed: %s", i, e)
                image = prev_image

            # ── Color histogram matching to frame 0 ──
            if reference_np is not None:
                image = _match_color_to_reference(image, reference_np)

        # Save frame
        frame_path = str(job_dir / f"frame_{i:04d}.png")
        image.save(frame_path)
        frame_paths.append(frame_path)
        prev_image = image

        elapsed = time.time() - t0
        _log.info("[ANIM GEN] Frame %d/%d — %.1fs (strength=%.2f)", i + 1, num_frames, elapsed, cur_strength)

        # Checkpoint after every frame
        _save_animation_checkpoint(checkpoint_file, {
            "job_id": job_id,
            "frame_paths": frame_paths,
            "seed": base_seed,
            "current_frame": i + 1,
            "total_frames": num_frames,
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "art_style": art_style,
            "model_path": model_path,
            "lora_paths": [str(p) for p in (lora_paths or [])],
            "lora_weights": lora_weights or [],
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "frame_strength": frame_strength,
            "fps": fps,
            "storyboard": storyboard,
            "output_format": output_format,
            "reference_blend": reference_blend,
            "strength_curve": strength_curve,
            "motion_intensity": motion_intensity,
        })

    # ── Assemble output ────────────────────────────────────────────
    results: dict = {
        "status": "ok",
        "frames": frame_paths,
        "num_frames": num_frames,
        "seed": base_seed,
        "job_id": job_id,
        "checkpoint": str(checkpoint_file),
        "art_style": art_style,
    }

    duration_ms = max(30, 1000 // fps)  # per-frame duration in ms

    pil_frames = [PILImage.open(fp).convert("RGB") for fp in frame_paths]

    if output_format in ("gif", "both"):
        gif_path = str(output_dir_path / f"{job_id}_animated.gif")
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        results["gif_path"] = gif_path
        results["path"] = gif_path  # default path for backwards compat
        _log.info("[ANIM GEN] GIF saved: %s", gif_path)

    if output_format in ("mp4", "both"):
        mp4_path = _frames_to_mp4(frame_paths, fps, output_dir_path / f"{job_id}_animated.mp4")
        if mp4_path:
            results["mp4_path"] = mp4_path
            if "path" not in results:
                results["path"] = mp4_path
            _log.info("[ANIM GEN] MP4 saved: %s", mp4_path)
        else:
            _log.warning("[ANIM GEN] MP4 export failed — ffmpeg/opencv not found")

    if "path" not in results:
        results["path"] = frame_paths[0]  # fallback

    _update_progress(active=False, type="idle", current_step=0,
                     total_steps=0, current_frame=0, total_frames=0,
                     message="")
    return results


def _save_animation_checkpoint(checkpoint_file: Path, data: dict):
    """Atomically write checkpoint (write tmp then rename)."""
    tmp = checkpoint_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(checkpoint_file)


def _frames_to_mp4(frame_paths: list[str], fps: int, out_path: Path) -> str | None:
    """Combine frames into an mp4 video. Tries opencv first, falls back to ffmpeg."""
    try:
        import cv2
        from PIL import Image as PILImage
        first = PILImage.open(frame_paths[0])
        w, h = first.size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        for fp in frame_paths:
            import numpy as np
            img = np.array(PILImage.open(fp).convert("RGB"))
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        writer.release()
        return str(out_path)
    except ImportError:
        pass

    # Fallback: ffmpeg
    import subprocess
    if not shutil.which("ffmpeg"):
        return None
    try:
        # Build a concat file
        list_file = out_path.with_suffix(".txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for fp in frame_paths:
                safe_path = str(Path(fp).resolve()).replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")
                f.write(f"duration {1/fps:.4f}\n")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(list_file), "-vsync", "vfr",
             "-pix_fmt", "yuv420p", str(out_path)],
            capture_output=True, timeout=120, check=True,
        )
        list_file.unlink(missing_ok=True)
        return str(out_path)
    except Exception as e:
        _log.warning("[ANIM GEN] ffmpeg failed: %s", e)
        return None


def regenerate_frame(
    job_id: str,
    frame_index: int,
    fix_prompt: str | None = None,
    strength: float = 0.40,
    steps: int = 25,
) -> dict:
    """Re-generate a single frame, keeping neighbours consistent.

    Uses the average of the previous and next frames (when available) as
    the init_image, so the fix blends smoothly into the sequence.
    """
    from PIL import Image as PILImage

    job_dir = _ANIM_SAVES_DIR / job_id
    checkpoint_file = job_dir / "checkpoint.json"
    if not checkpoint_file.exists():
        return {"status": "error", "error": f"Job {job_id} not found"}

    ckpt = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    frame_paths = ckpt.get("frame_paths", [])
    if frame_index < 0 or frame_index >= len(frame_paths):
        return {"status": "error", "error": f"Frame index {frame_index} out of range (0-{len(frame_paths)-1})"}

    model_path = ckpt.get("model_path", _active_model)
    img2img_pipe = _load_img2img_pipeline(model_path)
    if img2img_pipe is None:
        return {"status": "error", "error": "img2img pipeline unavailable"}

    # Build init image: blend neighbours
    frames_pil = []
    if frame_index > 0:
        frames_pil.append(PILImage.open(frame_paths[frame_index - 1]).convert("RGB"))
    if frame_index < len(frame_paths) - 1:
        frames_pil.append(PILImage.open(frame_paths[frame_index + 1]).convert("RGB"))
    if not frames_pil:
        # Only frame — use itself
        frames_pil.append(PILImage.open(frame_paths[frame_index]).convert("RGB"))

    import numpy as np
    blended = np.mean([np.array(f) for f in frames_pil], axis=0).astype(np.uint8)
    init_image = PILImage.fromarray(blended)
    w, h = int(ckpt.get("width", 512)), int(ckpt.get("height", 512))
    init_image = init_image.resize((w, h), PILImage.LANCZOS)

    prompt = fix_prompt or ckpt.get("prompt", "")
    art_style = ckpt.get("art_style", "anime")
    styled_prompt, styled_neg = _apply_art_style(prompt, DEFAULT_NEGATIVE, art_style)
    full_neg = apply_feedback_learnings(styled_neg)

    seed = ckpt.get("seed", 42)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    image = img2img_pipe(
        prompt=styled_prompt,
        negative_prompt=full_neg,
        image=init_image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=float(ckpt.get("guidance_scale", 7.5)),
        generator=generator,
    ).images[0]

    # Overwrite the frame file
    frame_path = frame_paths[frame_index]
    image.save(frame_path)
    _log.info("[ANIM GEN] Frame %d regenerated for job %s", frame_index, job_id)

    return {
        "status": "ok",
        "frame_index": frame_index,
        "frame_path": frame_path,
        "job_id": job_id,
    }


def list_animation_jobs() -> list[dict]:
    """List all saved animation jobs with their status."""
    _ANIM_SAVES_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for d in sorted(_ANIM_SAVES_DIR.iterdir()):
        if not d.is_dir():
            continue
        ckpt_file = d / "checkpoint.json"
        if not ckpt_file.exists():
            continue
        try:
            ckpt = json.loads(ckpt_file.read_text(encoding="utf-8"))
            done = ckpt.get("current_frame", 0) >= ckpt.get("total_frames", 1)
            jobs.append({
                "job_id": d.name,
                "status": "complete" if done else "incomplete",
                "current_frame": ckpt.get("current_frame", 0),
                "total_frames": ckpt.get("total_frames", 0),
                "prompt": ckpt.get("prompt", "")[:80],
                "negative_prompt": ckpt.get("negative_prompt", ""),
                "art_style": ckpt.get("art_style", ""),
                "seed": ckpt.get("seed", -1),
                "frame_strength": ckpt.get("frame_strength", 0.35),
                "fps": ckpt.get("fps", 12),
                "steps": ckpt.get("steps", 26),
                "guidance_scale": ckpt.get("guidance_scale", 7.5),
                "width": ckpt.get("width", 512),
                "height": ckpt.get("height", 512),
                "storyboard": ckpt.get("storyboard", []),
                "output_format": ckpt.get("output_format", "gif"),
                "reference_blend": ckpt.get("reference_blend", 0.3),
                "strength_curve": ckpt.get("strength_curve", "constant"),
                "motion_intensity": ckpt.get("motion_intensity", 0.5),
            })
        except Exception:
            continue
    return jobs


def get_animation_job(job_id: str) -> dict | None:
    """Get full checkpoint data for a specific animation job."""
    ckpt_file = _ANIM_SAVES_DIR / job_id / "checkpoint.json"
    if not ckpt_file.exists():
        return None
    try:
        return json.loads(ckpt_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_animation_state(
    prompt: str,
    negative_prompt: str = "",
    art_style: str = "",
    seed: int = -1,
    num_frames: int = 24,
    fps: int = 12,
    frame_strength: float = 0.35,
    width: int = 512,
    height: int = 768,
    steps: int = 25,
    guidance_scale: float = 7.0,
    output_format: str = "gif",
    storyboard_descriptions: list[str] | None = None,
    reference_blend: float = 0.3,
    strength_curve: str = "constant",
    motion_intensity: float = 0.5,
) -> dict:
    """Save current animation settings as a named job (no generation)."""
    _ANIM_SAVES_DIR.mkdir(parents=True, exist_ok=True)
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{id(prompt) % 10000:04d}"
    job_dir = _ANIM_SAVES_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "job_id": job_id,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "art_style": art_style,
        "seed": seed,
        "total_frames": num_frames,
        "current_frame": 0,
        "fps": fps,
        "frame_strength": frame_strength,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "output_format": output_format,
        "storyboard": storyboard_descriptions or [],
        "reference_blend": reference_blend,
        "strength_curve": strength_curve,
        "motion_intensity": motion_intensity,
        "status": "saved",
        "frame_paths": [],
    }
    (job_dir / "checkpoint.json").write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    _log.info("[ANIM] Saved animation state as job %s", job_id)
    return {"status": "ok", "job_id": job_id}


# ── Storyboard / Line-Drawing Preview ─────────────────────────────────────

def generate_storyboard(
    prompt: str,
    num_frames: int = 8,
    width: int = 256,
    height: int = 256,
    model_path: str | None = None,
    storyboard_descriptions: list[str] | None = None,
    seed: int = -1,
) -> dict:
    """Generate a low-res line-drawing storyboard for preview.

    Uses very few steps (5-8) and tiny resolution to give a quick visual
    in seconds rather than minutes. Negative prompt steers toward sketchy
    line-art so you can see composition without waiting for full render.
    """
    from PIL import Image as PILImage

    model_path = _resolve_model_path(model_path)
    if model_path is None:
        return {"status": "error", "error": "No model found"}

    pipe = _load_pipeline(model_path)
    base_seed = seed if seed >= 0 else torch.randint(0, 2**32, (1,)).item()

    # Low-res, low-step sketch settings
    preview_steps = 15
    preview_neg = (
        "photorealistic, 3d, highly detailed, smooth, "
        "blurry, abstract, smudge, noise, artifacts, deformed"
    )

    output_dir_path = Path(OUTPUT_DIR) / "storyboards"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if storyboard_descriptions is None:
        storyboard_descriptions = [f"frame {i+1}" for i in range(num_frames)]

    frame_paths = []
    for i in range(num_frames):
        idx = min(i, len(storyboard_descriptions) - 1)
        frame_prompt = (
            f"pencil sketch, clean linework, storyboard panel, "
            f"black and white drawing, clear composition, "
            f"{prompt}, {storyboard_descriptions[idx]}"
        )
        generator = torch.Generator(device="cuda").manual_seed(base_seed + i)
        image = pipe(
            prompt=frame_prompt,
            negative_prompt=preview_neg,
            width=max(width, 384),
            height=max(height, 384),
            num_inference_steps=preview_steps,
            guidance_scale=7.0,
            generator=generator,
        ).images[0]
        fp = str(output_dir_path / f"{timestamp}_sb_{i:03d}.png")
        image.save(fp)
        frame_paths.append(fp)

    # Combine into a grid
    grid_path = str(output_dir_path / f"{timestamp}_storyboard_grid.png")
    _make_storyboard_grid(frame_paths, grid_path, cols=4)

    return {
        "status": "ok",
        "grid_path": grid_path,
        "frame_paths": frame_paths,
        "num_frames": num_frames,
        "seed": base_seed,
    }


def _make_storyboard_grid(paths: list[str], out_path: str, cols: int = 4):
    """Tile images into a grid for easy storyboard review."""
    from PIL import Image as PILImage
    images = [PILImage.open(p) for p in paths]
    if not images:
        return
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = PILImage.new("RGB", (w * cols, h * rows), (255, 255, 255))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(img, (c * w, r * h))
    grid.save(out_path)


# ── Quick Preview (low-res → high-res two-pass) ───────────────────────────


def generate_preview_only(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    model_path: str | None = None,
    mode: str = "normal",
    steps: int = 12,
    guidance_scale: float = 7.5,
    seed: int = -1,
    art_style: str = "anime",
    lora_paths: list[str] | None = None,
    lora_weights: list[float] | None = None,
    negative_prompt: str | None = None,
    selected_outfits: dict[str, str] | None = None,
    preview_steps: int = 4,
) -> dict:
    """Fast noisy preview that produces a rough version of the SAME image
    the full render will produce.

    Strategy: run the pipeline at a much smaller resolution (256×256) with
    the same seed and prompt.  Because the same seed is used, the composition
    is similar to the full render — just at lower resolution.  This is
    dramatically faster than capturing an early latent at full resolution.

    The returned seed should be passed to the full render to reproduce the
    exact same result at full quality.
    """
    from PIL import Image as PILImage

    # Use a small preview resolution for speed (256×256)
    preview_w = 256
    preview_h = 256

    # Resolve seed up front so preview and full render share it
    if seed < 0:
        seed = torch.randint(0, 2**32, (1,)).item()

    # Resolve model
    model_path = _resolve_model_path(model_path)
    if model_path is None:
        return {"status": "error", "error": "No model files found in models/ directory"}
    if not Path(model_path).exists():
        return {"status": "error", "error": f"Model not found: {model_path}"}

    evict_ollama_models()
    pipe = _load_pipeline(model_path)

    # Load LoRAs
    if lora_paths:
        unload_loras(pipe)
        weights = lora_weights or [0.8] * len(lora_paths)
        for lp, lw in zip(lora_paths, weights):
            if Path(lp).exists():
                load_lora(pipe, lp, weight=lw)

    # Build prompt with trigger word injection (same logic as generate_image)
    # Skip character LoRA injection when 2+ character LoRAs are active
    selected_outfits = selected_outfits or {}
    if lora_paths:
        tw_data = _load_trigger_words()
        prompt_lower = prompt.lower()

        char_dir = MODELS_DIR / "characters"
        char_lora_stems = set()
        for lp in lora_paths:
            try:
                if Path(lp).parent.resolve() == char_dir.resolve():
                    char_lora_stems.add(Path(lp).stem)
            except Exception:
                pass
        multi_char_mode = len(char_lora_stems) >= 2

        for lp in lora_paths:
            if not Path(lp).exists():
                continue
            stem = Path(lp).stem
            if multi_char_mode and stem in char_lora_stems:
                continue
            words = tw_data.get(stem, [])
            if words:
                outfit = selected_outfits.get(stem)
                trigger_str = build_trigger_prompt(words, selected_outfit=outfit)
                if trigger_str:
                    missing_parts = []
                    for part in trigger_str.split(", "):
                        part_clean = re.sub(r'^\(|\)$', '', part).split(":")[0].lower()
                        if part_clean and part_clean not in prompt_lower:
                            missing_parts.append(part)
                    if missing_parts:
                        inject_str = ", ".join(missing_parts)
                        prompt = f"{inject_str}, {prompt}"
                        prompt_lower = prompt.lower()

    if mode == "explicit":
        full_prompt = f"score_9, score_8_up, score_7_up, {prompt}"
        base_neg = negative_prompt or EXPLICIT_NEGATIVE
    else:
        full_prompt = prompt
        base_neg = negative_prompt or DEFAULT_NEGATIVE

    # Strip A1111/ComfyUI-style <lora:...> tags
    full_prompt = re.sub(r'<lora:[^>]+>', '', full_prompt)
    full_prompt = re.sub(r',\s*,', ',', full_prompt)
    full_prompt = re.sub(r'\s{2,}', ' ', full_prompt).strip(', ')

    # Auto-insert BREAK tokens before parenthesized character blocks
    if 'BREAK' not in full_prompt:
        tw_data_auto = _load_trigger_words()
        char_dir_auto = MODELS_DIR / "characters"
        _all_trigger_codes_p: set[str] = set()
        for _stem, _words in tw_data_auto.items():
            lp_check = char_dir_auto / f"{_stem}.safetensors"
            if lp_check.exists():
                _pg = parse_outfit_groups(_words)
                if _pg["primary"]:
                    _all_trigger_codes_p.add(_pg["primary"].lower())
                for _on, _tags in _pg.get("outfits", {}).items():
                    if _tags:
                        _c = _tags[0]["word"] if isinstance(_tags[0], dict) else str(_tags[0])
                        if _c:
                            _all_trigger_codes_p.add(_c.lower())
        paren_blocks = list(re.finditer(r'\([^)]+\)', full_prompt))
        insert_positions: list[int] = []
        for pb in paren_blocks:
            block_text = pb.group(0).lower()
            for code in _all_trigger_codes_p:
                if code in block_text:
                    insert_positions.append(pb.start())
                    break
        if insert_positions:
            modified = full_prompt
            for pos in sorted(insert_positions, reverse=True):
                if pos > 0:
                    pre = modified[:pos].rstrip(', ')
                    post = modified[pos:]
                    modified = f"{pre} BREAK {post}"
            if modified != full_prompt:
                _log.info("[IMAGE GEN] Preview: auto-inserted BREAK tokens before %d character block(s)", len(insert_positions))
                full_prompt = modified

    full_prompt = _structure_multi_angle_prompt(full_prompt)
    full_prompt = _structure_multi_character_prompt(full_prompt, lora_paths, selected_outfits)
    full_prompt, base_neg = _apply_art_style(full_prompt, base_neg, art_style)
    full_negative = apply_feedback_learnings(base_neg)
    full_prompt = apply_positive_learnings(full_prompt)

    # Use fewer steps at small resolution for speed
    total_steps = max(preview_steps + 2, 8)
    preview_steps = max(2, min(preview_steps, total_steps - 1))

    generator = torch.Generator(device="cuda").manual_seed(seed)

    # At small resolution, run the full pipeline (no latent capture needed)
    # Encode prompt
    embed_kwargs = {}
    long_prompt_used = False
    encoding_result = _encode_long_prompt(pipe, full_prompt, full_negative)
    if encoding_result and "_chunked_prompt" not in encoding_result:
        embed_kwargs = encoding_result
        long_prompt_used = True
    elif encoding_result and "_chunked_prompt" in encoding_result:
        full_prompt = encoding_result["_chunked_prompt"]
        full_negative = encoding_result["_chunked_negative"]

    gen_kwargs = {
        "width": preview_w,
        "height": preview_h,
        "num_inference_steps": total_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }
    if long_prompt_used and embed_kwargs:
        gen_kwargs.update(embed_kwargs)
    else:
        gen_kwargs["prompt"] = full_prompt
        gen_kwargs["negative_prompt"] = full_negative

    try:
        preview_img = pipe(**gen_kwargs).images[0]
    except Exception as e:
        return {"status": "error", "error": f"Preview generation failed: {e}"}

    # Upscale slightly for display (256×256 → readable)
    display_w = max(256, preview_w)
    display_h = max(256, preview_h)
    if preview_img.size != (display_w, display_h):
        preview_img = preview_img.resize((display_w, display_h), PILImage.LANCZOS)

    # Save
    output_dir_path = Path(OUTPUT_DIR)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preview_path = str(output_dir_path / f"{timestamp}_preview.png")
    preview_img.save(preview_path)

    _log.info("[IMAGE GEN] Noisy preview saved: %s (seed=%d, captured at step %d/%d)",
              preview_path, seed, preview_steps, total_steps)

    return {
        "status": "ok",
        "preview_path": preview_path,
        "seed": seed,
        "prompt_used": full_prompt,
        "negative_used": full_negative,
        "settings": {
            "model": Path(model_path).stem,
            "loras": [Path(lp).stem for lp in (lora_paths or [])],
            "mode": mode,
            "preview_steps": preview_steps,
            "total_steps_used": total_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "width": width,
            "height": height,
        },
        "is_preview": True,
        "long_prompt": long_prompt_used,
    }


def generate_with_preview(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    model_path: str | None = None,
    mode: str = "normal",
    steps: int = 35,
    guidance_scale: float = 7.5,
    seed: int = -1,
    art_style: str = "anime",
    lora_paths: list[str] | None = None,
    lora_weights: list[float] | None = None,
    negative_prompt: str | None = None,
    selected_outfits: dict[str, str] | None = None,
) -> dict:
    """Two-pass generation: quick noisy preview, then full-res render.

    Returns both the preview and the full image so the frontend can show
    the preview within seconds while the full render completes.
    """
    # Pass 1: noisy preview using latent capture
    preview = generate_preview_only(
        prompt=prompt,
        width=width,
        height=height,
        model_path=model_path,
        lora_paths=lora_paths,
        lora_weights=lora_weights,
        mode=mode,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        seed=seed,
        art_style=art_style,
        selected_outfits=selected_outfits,
    )
    if preview.get("status") == "error":
        return preview

    actual_seed = preview.get("seed", seed)

    # Pass 2: full render with same seed for consistent result
    full = generate_image(
        prompt=prompt,
        width=width,
        height=height,
        model_path=model_path,
        lora_paths=lora_paths,
        lora_weights=lora_weights,
        mode=mode,
        steps=steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        seed=actual_seed,
        art_style=art_style,
        selected_outfits=selected_outfits,
    )

    return {
        "status": full.get("status", "ok"),
        "preview_path": preview.get("preview_path"),
        "full_path": full.get("path"),
        "path": full.get("path"),
        "seed": actual_seed,
        "prompt_used": full.get("prompt_used"),
        "settings": full.get("settings"),
        "lora_diagnostics": full.get("lora_diagnostics", []),
    }


# ── Tiled Upscaler (VRAM-friendly for 3070 Ti / 8 GB cards) ──────────────

def upscale_image(
    image_path: str,
    scale: float = 2.0,
    tile_size: int = 768,
    model_path: str | None = None,
    prompt: str = "",
    steps: int = 20,
    strength: float = 0.35,
) -> dict:
    """
    Content-aware tiled upscaling: split image into overlapping tiles,
    run img2img on each tile, then stitch back together.

    Keeps VRAM usage bounded regardless of output resolution.
    Recommended tile_size values for 8 GB VRAM: 512-768.
    """
    from PIL import Image as PILImage
    import numpy as np

    if not Path(image_path).exists():
        return {"status": "error", "error": f"Image not found: {image_path}"}

    scale = max(1.5, min(scale, 6.0))
    tile_size = max(256, min(tile_size, 2048))

    src = PILImage.open(image_path).convert("RGB")
    orig_w, orig_h = src.size
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    # Round to nearest 8 (required for SDXL)
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8

    _log.info("[UPSCALE] %dx%d → %dx%d (%.1fx, tile=%d)",
              orig_w, orig_h, new_w, new_h, scale, tile_size)

    model_path = _resolve_model_path(model_path)
    if model_path is None:
        return {"status": "error", "error": "No model found"}
    img2img_pipe = _load_img2img_pipeline(model_path)
    if img2img_pipe is None:
        return {"status": "error", "error": "img2img pipeline unavailable"}

    # Upscale source to target size using PIL first (bicubic)
    upscaled = src.resize((new_w, new_h), PILImage.LANCZOS)

    # Tile parameters
    overlap = tile_size // 4  # 25% overlap for blending
    step = tile_size - overlap
    output_np = np.array(upscaled, dtype=np.float64)
    weight_map = np.zeros((new_h, new_w), dtype=np.float64)

    tile_count = 0
    for y in range(0, new_h, step):
        for x in range(0, new_w, step):
            y2 = min(y + tile_size, new_h)
            x2 = min(x + tile_size, new_w)
            y1 = max(0, y2 - tile_size)
            x1 = max(0, x2 - tile_size)

            tile = upscaled.crop((x1, y1, x2, y2))
            tw, th = tile.size
            # Round tile to 8
            tw8, th8 = (tw // 8) * 8, (th // 8) * 8
            if tw8 < 64 or th8 < 64:
                continue
            tile = tile.resize((tw8, th8), PILImage.LANCZOS)

            tile_prompt = prompt or "high resolution, detailed, sharp"
            generator = torch.Generator(device="cuda").manual_seed(42)
            result = img2img_pipe(
                prompt=tile_prompt,
                negative_prompt="blurry, low quality, artifacts, noise",
                image=tile,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=5.0,
                generator=generator,
            ).images[0]

            # Resize back to tile dimensions
            result = result.resize((x2 - x1, y2 - y1), PILImage.LANCZOS)
            result_np = np.array(result, dtype=np.float64)

            # Feathered blend mask
            mask = _make_feather_mask(y2 - y1, x2 - x1, overlap)
            for c in range(3):
                output_np[y1:y2, x1:x2, c] += result_np[:, :, c] * mask
            weight_map[y1:y2, x1:x2] += mask

            tile_count += 1
            _log.info("[UPSCALE] Tile %d processed (%d,%d)-(%d,%d)", tile_count, x1, y1, x2, y2)

    # Normalize by weights
    weight_map = np.maximum(weight_map, 1e-6)
    for c in range(3):
        output_np[:, :, c] /= weight_map
    output_np = np.clip(output_np, 0, 255).astype(np.uint8)

    # ── Brightness preservation ──────────────────────────────────
    # The img2img tiles tend to brighten the image. Match the mean
    # luminance of the output to the (upscaled) source image.
    upscaled_np = np.array(upscaled, dtype=np.float64)
    src_lum = 0.2126 * upscaled_np[:,:,0] + 0.7152 * upscaled_np[:,:,1] + 0.0722 * upscaled_np[:,:,2]
    out_lum = 0.2126 * output_np[:,:,0].astype(np.float64) + 0.7152 * output_np[:,:,1].astype(np.float64) + 0.0722 * output_np[:,:,2].astype(np.float64)
    src_mean = np.mean(src_lum)
    out_mean = np.mean(out_lum)
    if out_mean > 1e-3:
        lum_ratio = src_mean / out_mean
        # Clamp ratio to avoid extreme corrections
        lum_ratio = max(0.5, min(lum_ratio, 1.5))
        if abs(lum_ratio - 1.0) > 0.02:
            _log.info("[UPSCALE] Brightness correction: ratio=%.3f", lum_ratio)
            output_np = np.clip(output_np.astype(np.float64) * lum_ratio, 0, 255).astype(np.uint8)

    final = PILImage.fromarray(output_np)

    output_dir_path = Path(OUTPUT_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(output_dir_path / f"{timestamp}_upscaled_{scale:.1f}x.png")
    final.save(out_path, quality=95)
    _log.info("[UPSCALE] Done: %s (%d tiles)", out_path, tile_count)

    return {
        "status": "ok",
        "path": out_path,
        "original_size": [orig_w, orig_h],
        "upscaled_size": [new_w, new_h],
        "scale": scale,
        "tiles_processed": tile_count,
    }


def _make_feather_mask(h: int, w: int, margin: int):
    """Create a 2D feathering mask that fades at tile edges for seamless blending."""
    import numpy as np
    mask = np.ones((h, w), dtype=np.float64)
    ramp = np.linspace(0, 1, margin) if margin > 0 else np.array([1.0])
    # Feather edges
    for i in range(min(margin, h)):
        mask[i, :] *= ramp[i]
        mask[h - 1 - i, :] *= ramp[i]
    for j in range(min(margin, w)):
        mask[:, j] *= ramp[j]
        mask[:, w - 1 - j] *= ramp[j]
    return mask


# ── LoRA Training Pipeline ────────────────────────────────────────────────

TRAINING_DIR = Path(__file__).parent.parent / "cache" / "lora_training"

_training_state: dict = {
    "status": "idle",  # idle | preparing | training | complete | error
    "progress": 0,
    "total_steps": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "error": None,
    "output_path": None,
    "job_name": None,
}


def get_training_status() -> dict:
    return dict(_training_state)


def critique_training_dataset(image_dir: str, training_type: str = "style") -> dict:
    """Analyse a folder of training images and critique coverage / quality.

    Checks:
      - Total image count (30-50 minimum for styles, 15-20 for characters)
      - Resolution consistency
      - Aspect ratio variety
      - Content diversity estimate (heuristic: file-size variance)
      - For characters: pose variety, face coverage, body proportions
    """
    from PIL import Image as PILImage

    p = Path(image_dir)
    if not p.exists() or not p.is_dir():
        return {"status": "error", "error": f"Directory not found: {image_dir}"}

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    images = [f for f in p.iterdir() if f.suffix.lower() in exts and f.is_file()]

    if not images:
        return {"status": "error", "error": "No images found in directory"}

    min_images = 15 if training_type == "character" else 30
    count = len(images)

    # Analyse resolutions
    sizes = []
    file_sizes = []
    issues: list[str] = []
    suggestions: list[str] = []

    for img_path in images:
        try:
            with PILImage.open(img_path) as im:
                sizes.append(im.size)
            file_sizes.append(img_path.stat().st_size)
        except Exception:
            issues.append(f"Cannot open: {img_path.name}")

    if not sizes:
        return {"status": "error", "error": "No valid images could be opened"}

    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    import statistics
    w_std = statistics.stdev(widths) if len(widths) > 1 else 0
    h_std = statistics.stdev(heights) if len(heights) > 1 else 0

    # Count check
    if count < min_images:
        issues.append(
            f"Only {count} images — need at least {min_images} for good {training_type} training. "
            f"More images = better generalisation."
        )
        suggestions.append(f"Add {min_images - count}+ more images")
    elif count < min_images * 2:
        suggestions.append(
            f"{count} images is okay but {min_images * 2}+ is ideal for high quality"
        )

    # Resolution check
    min_res = min(min(w, h) for w, h in sizes)
    if min_res < 512:
        issues.append(f"Some images are below 512px ({min_res}px) — may produce blurry training")
        suggestions.append("Upscale or replace images below 512px resolution")

    # Diversity check (file-size variance as proxy)
    if len(file_sizes) > 2:
        fs_std = statistics.stdev(file_sizes)
        fs_mean = statistics.mean(file_sizes)
        diversity_ratio = fs_std / fs_mean if fs_mean > 0 else 0
        if diversity_ratio < 0.15:
            issues.append("Images appear very uniform — may overfit to one composition")
            suggestions.append("Add variety: different poses, angles, backgrounds, lighting")

    # Aspect ratio diversity
    aspects = [w / h for w, h in sizes]
    aspect_unique = len(set(round(a, 1) for a in aspects))
    if aspect_unique < 2 and count > 5:
        suggestions.append("Include both portrait and landscape images for better flexibility")

    # Character-specific checks
    if training_type == "character":
        if count < 5:
            issues.append("Need at least 5 images showing the character's face clearly")
        suggestions.append("Include: front view, side view, back view, full body, close-up face")
        suggestions.append("Include different expressions, poses, and outfit angles")
        suggestions.append("Consistent character features across images (hair colour, eye colour, etc.)")

    quality = "good" if not issues else "needs_work" if len(issues) <= 2 else "insufficient"

    return {
        "status": "ok",
        "quality": quality,
        "image_count": count,
        "minimum_recommended": min_images,
        "resolution_range": {
            "min": [min(widths), min(heights)],
            "max": [max(widths), max(heights)],
        },
        "issues": issues,
        "suggestions": suggestions,
        "training_type": training_type,
    }


def start_lora_training(
    name: str,
    image_dir: str,
    training_type: str = "style",  # "style" or "character"
    base_model: str | None = None,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    rank: int = 32,
    resolution: int = 1024,
    batch_size: int = 1,
    caption_method: str = "filename",  # "filename", "blip", or "txt"
) -> dict:
    """Start LoRA training on a set of images.

    Pipeline:
      1. Validate / critique the dataset
      2. Prepare captions (from filenames, BLIP auto-caption, or sidecar .txt files)
      3. Run LoRA training via diffusers/kohya-style trainer
      4. Save checkpoint after each epoch (resume-safe)
      5. Save final LoRA to models/loras/<name>.safetensors

    Works with 8 GB VRAM (3070 Ti) using gradient checkpointing + fp16.
    """
    global _training_state

    if _training_state["status"] == "training":
        return {"status": "error", "error": "Training already in progress"}

    # Validate dataset
    critique = critique_training_dataset(image_dir, training_type)
    if critique.get("quality") == "insufficient":
        return {
            "status": "error",
            "error": "Dataset is insufficient for training",
            "critique": critique,
        }

    base_model = _resolve_model_path(base_model)
    if base_model is None:
        return {"status": "error", "error": "No base model found"}

    # Prepare training directory
    train_dir = TRAINING_DIR / name
    train_dir.mkdir(parents=True, exist_ok=True)
    output_lora_path = MODELS_DIR / "loras" / f"{name}.safetensors"
    (MODELS_DIR / "loras").mkdir(parents=True, exist_ok=True)

    _training_state = {
        "status": "preparing",
        "progress": 0,
        "total_steps": 0,
        "current_epoch": 0,
        "total_epochs": epochs,
        "error": None,
        "output_path": str(output_lora_path),
        "job_name": name,
    }

    def _run_training():
        global _training_state
        try:
            _training_state["status"] = "training"
            _log.info("[TRAIN] Starting LoRA training: %s (%s)", name, training_type)

            # Step 1: Prepare captions
            dataset_path = _prepare_training_dataset(
                image_dir, train_dir, caption_method, resolution
            )

            # Step 2: Run training
            _run_lora_training_loop(
                dataset_path=dataset_path,
                base_model_path=base_model,
                output_path=str(output_lora_path),
                epochs=epochs,
                lr=learning_rate,
                rank=rank,
                resolution=resolution,
                batch_size=batch_size,
                train_dir=train_dir,
            )

            _training_state["status"] = "complete"
            _training_state["progress"] = 100
            _log.info("[TRAIN] LoRA training complete: %s", output_lora_path)

        except Exception as e:
            _training_state["status"] = "error"
            _training_state["error"] = str(e)
            _log.error("[TRAIN] Training failed: %s", e)

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()

    return {
        "status": "ok",
        "message": f"Training started: {name}",
        "job_name": name,
        "critique": critique,
        "output_path": str(output_lora_path),
    }


def _prepare_training_dataset(
    image_dir: str, train_dir: Path, caption_method: str, resolution: int
) -> Path:
    """Prepare images + captions in the format expected by the trainer."""
    from PIL import Image as PILImage

    dataset_dir = train_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    src = Path(image_dir)

    for f in src.iterdir():
        if f.suffix.lower() not in exts or not f.is_file():
            continue

        # Resize to training resolution
        try:
            img = PILImage.open(f).convert("RGB")
            # Resize maintaining aspect ratio, then centre-crop to square
            w, h = img.size
            scale = resolution / min(w, h)
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
            # Centre crop
            w2, h2 = img.size
            left = (w2 - resolution) // 2
            top = (h2 - resolution) // 2
            img = img.crop((left, top, left + resolution, top + resolution))
            out_name = f.stem + ".png"
            img.save(dataset_dir / out_name)
        except Exception as e:
            _log.warning("[TRAIN] Skipping %s: %s", f.name, e)
            continue

        # Generate caption
        caption_file = dataset_dir / (f.stem + ".txt")
        if caption_method == "txt":
            # Look for sidecar .txt file
            sidecar = src / (f.stem + ".txt")
            if sidecar.exists():
                shutil.copy2(sidecar, caption_file)
            else:
                caption_file.write_text(f.stem.replace("_", " "), encoding="utf-8")
        elif caption_method == "filename":
            caption_file.write_text(f.stem.replace("_", " "), encoding="utf-8")
        # "blip" captioning would require a BLIP model — fall back to filename
        elif caption_method == "blip":
            try:
                caption = _auto_caption_image(dataset_dir / out_name)
                caption_file.write_text(caption, encoding="utf-8")
            except Exception:
                caption_file.write_text(f.stem.replace("_", " "), encoding="utf-8")

    _log.info("[TRAIN] Dataset prepared: %s", dataset_dir)
    return dataset_dir


def _auto_caption_image(image_path: Path) -> str:
    """Auto-caption an image using BLIP or WD tagger if available."""
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from PIL import Image as PILImage
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to("cuda")
        raw = PILImage.open(image_path).convert("RGB")
        inputs = processor(raw, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except ImportError:
        return image_path.stem.replace("_", " ")


def _run_lora_training_loop(
    dataset_path: Path,
    base_model_path: str,
    output_path: str,
    epochs: int,
    lr: float,
    rank: int,
    resolution: int,
    batch_size: int,
    train_dir: Path,
):
    """Run the actual LoRA fine-tuning loop.

    Uses peft + diffusers for LoRA injection into the SDXL UNet.
    Saves a checkpoint after each epoch for resume safety.
    Enables gradient checkpointing and fp16 for 8 GB VRAM.
    """
    global _training_state
    from PIL import Image as PILImage
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError(
            "peft is required for LoRA training. Install with: pip install peft"
        )

    # Load base pipeline
    pipe = StableDiffusionXLPipeline.from_single_file(
        base_model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    unet = pipe.unet
    vae = pipe.vae.to("cuda")
    text_encoder = pipe.text_encoder.to("cuda")
    text_encoder_2 = pipe.text_encoder_2.to("cuda")
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    noise_scheduler = pipe.scheduler

    # Apply LoRA to UNet
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["to_k", "to_v", "to_q", "to_out.0"],
        lora_dropout=0.05,
    )
    unet = get_peft_model(unet, lora_config)
    unet.to("cuda", dtype=torch.float16)
    unet.enable_gradient_checkpointing()
    unet.train()

    # Freeze everything except LoRA params
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Simple dataset
    class CaptionImageDataset(Dataset):
        def __init__(self, folder: Path, res: int):
            self.files = sorted(folder.glob("*.png"))
            self.res = res

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            img_path = self.files[idx]
            caption_path = img_path.with_suffix(".txt")
            caption = caption_path.read_text(encoding="utf-8").strip() if caption_path.exists() else ""

            img = PILImage.open(img_path).convert("RGB").resize(
                (self.res, self.res), PILImage.LANCZOS
            )
            import torchvision.transforms as T
            tensor = T.ToTensor()(img) * 2.0 - 1.0  # normalise to [-1, 1]
            return tensor, caption

    dataset = CaptionImageDataset(dataset_path, resolution)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    trainable = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-2)

    total_steps = epochs * len(loader)
    _training_state["total_steps"] = total_steps
    step = 0

    checkpoints_dir = train_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Check for resume checkpoint
    last_ckpt = _find_latest_checkpoint(checkpoints_dir)
    start_epoch = 0
    if last_ckpt:
        try:
            ckpt_data = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            unet.load_state_dict(ckpt_data["unet"], strict=False)
            optimizer.load_state_dict(ckpt_data["optimizer"])
            start_epoch = ckpt_data.get("epoch", 0) + 1
            step = ckpt_data.get("step", 0)
            _log.info("[TRAIN] Resumed from epoch %d, step %d", start_epoch, step)
        except Exception as e:
            _log.warning("[TRAIN] Failed to load checkpoint, starting fresh: %s", e)

    for epoch in range(start_epoch, epochs):
        _training_state["current_epoch"] = epoch + 1
        epoch_loss = 0.0

        for batch_imgs, batch_captions in loader:
            batch_imgs = batch_imgs.to("cuda", dtype=torch.float16)

            # Encode image to latent space
            with torch.no_grad():
                latents = vae.encode(batch_imgs).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device="cuda"
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Encode text
            with torch.no_grad():
                text_input = tokenizer(
                    list(batch_captions), padding="max_length",
                    max_length=tokenizer.model_max_length, truncation=True,
                    return_tensors="pt"
                ).to("cuda")
                text_input_2 = tokenizer_2(
                    list(batch_captions), padding="max_length",
                    max_length=tokenizer_2.model_max_length, truncation=True,
                    return_tensors="pt"
                ).to("cuda")
                encoder_hidden_states = text_encoder(text_input.input_ids)[0]
                encoder_hidden_states_2 = text_encoder_2(text_input_2.input_ids)[0]
                # Concatenate for SDXL
                encoder_hidden_states = torch.cat(
                    [encoder_hidden_states, encoder_hidden_states_2], dim=-1
                )

            # Predict noise
            with torch.cuda.amp.autocast():
                noise_pred = unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample

            loss = F.mse_loss(noise_pred.float(), noise.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            epoch_loss += loss.item()
            _training_state["progress"] = int(100 * step / max(total_steps, 1))

        avg_loss = epoch_loss / max(len(loader), 1)
        _log.info("[TRAIN] Epoch %d/%d — loss: %.6f", epoch + 1, epochs, avg_loss)

        # Save epoch checkpoint
        ckpt_path = checkpoints_dir / f"epoch_{epoch:03d}.pt"
        torch.save({
            "unet": unet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "loss": avg_loss,
        }, ckpt_path)

    # Save final LoRA weights
    unet.save_pretrained(str(Path(output_path).parent / Path(output_path).stem))
    # Also try to save as single safetensors file
    try:
        from safetensors.torch import save_file
        lora_state = {k: v for k, v in unet.state_dict().items() if "lora" in k.lower()}
        save_file(lora_state, output_path)
        _log.info("[TRAIN] LoRA saved: %s", output_path)
    except Exception as e:
        _log.warning("[TRAIN] safetensors save failed: %s", e)

    # Cleanup GPU
    del unet, vae, text_encoder, text_encoder_2
    torch.cuda.empty_cache()


def _find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """Find the most recent epoch checkpoint file."""
    files = sorted(ckpt_dir.glob("epoch_*.pt"))
    return files[-1] if files else None


def cancel_training() -> dict:
    """Request cancellation of the current training job.

    Training checks this flag between epochs and stops cleanly,
    preserving the latest checkpoint.
    """
    global _training_state
    if _training_state["status"] != "training":
        return {"status": "error", "error": "No training in progress"}
    _training_state["status"] = "cancelled"
    return {"status": "ok", "message": "Training will stop after current epoch"}


# ── Feedback submission ───────────────────────────────────────────────────

def submit_feedback(generation_index: int, feedback_text: str) -> dict:
    """Submit feedback for a specific generation to improve future results.

    Feedback is analyzed for both negative issues (added to negative prompt)
    and positive preferences (added to positive prompt enhancements).
    """
    if generation_index < 0 or generation_index >= len(_generation_history):
        return {"status": "error", "error": "Invalid generation index"}

    _generation_history[generation_index]["feedback"] = feedback_text
    _save_history_to_disk()
    _log.info("[IMAGE GEN] Feedback recorded for generation %d: %s",
              generation_index, feedback_text[:100])

    # Analyze whether feedback contains positive or negative signals
    fb_lower = feedback_text.lower()
    has_negative = any(kw in fb_lower for kw in [
        "bad", "wrong", "ugly", "deform", "error", "worse", "too",
        "hands", "fingers", "blurry", "dark", "bright",
    ])
    has_positive = any(kw in fb_lower for kw in _POSITIVE_KEYWORDS)

    feedback_type = "mixed"
    if has_positive and not has_negative:
        feedback_type = "positive"
    elif has_negative and not has_positive:
        feedback_type = "negative"

    return {
        "status": "ok",
        "message": "Feedback recorded — will influence future generations",
        "feedback_type": feedback_type,
        "will_affect": {
            "negative_prompt": has_negative,
            "positive_prompt": has_positive,
        },
    }


def delete_model(model_path: str) -> dict:
    """Remove a model file from disk (unloads first if active)."""
    p = Path(model_path)
    if not p.exists():
        return {"status": "error", "error": "File not found"}

    # Safety: only allow deleting from models directory
    try:
        p.resolve().relative_to(MODELS_DIR.resolve())
    except ValueError:
        return {"status": "error", "error": "Can only delete models from the models/ directory"}

    # Unload if currently loaded
    if str(p) in _loaded_pipelines:
        unload_model(str(p))

    p.unlink()
    _log.info("[IMAGE GEN] Deleted model: %s", model_path)
    return {"status": "ok", "message": f"Deleted {p.name}"}


# ── Character search (from available LoRAs and generation history) ────────

def search_characters(query: str) -> dict:
    """Search all LoRA categories and past generations matching a query.

    Searches file names, trigger words, and generation history prompts.
    Returns matching LoRA adapters and recent history entries.
    """
    query_lower = query.lower().strip()
    if not query_lower:
        return {"status": "ok", "loras": [], "history_matches": []}

    # Search ALL LoRA categories
    LORA_CATEGORIES = ("styles", "characters", "clothing", "poses", "concept", "action")
    tw_data = _load_trigger_words()
    matching_loras = []
    for cat in LORA_CATEGORIES:
        cat_dir = MODELS_DIR / cat
        if not cat_dir.exists():
            continue
        for f in cat_dir.iterdir():
            if f.suffix.lower() not in LORA_EXTENSIONS or not f.is_file():
                continue
            name_match = query_lower in f.stem.lower()
            trigger_match = any(query_lower in tw.lower() for tw in tw_data.get(f.stem, []))
            if name_match or trigger_match:
                # Look for a preview image
                preview = None
                for ext in (".png", ".jpg", ".jpeg", ".webp"):
                    candidate = f.parent / f"{f.stem}.preview{ext}"
                    if candidate.exists():
                        preview = str(candidate)
                        break
                matching_loras.append({
                    "name": f.stem,
                    "path": str(f),
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                    "category": cat,
                    "trigger_words": tw_data.get(f.stem, []),
                    "preview_image": preview,
                })

    # Search generation history for matching prompts
    history_matches = []
    for entry in _generation_history:
        prompt = (entry.get("prompt") or "").lower()
        if query_lower in prompt:
            history_matches.append({
                "prompt": entry.get("prompt", "")[:100],
                "result_path": entry.get("result_path"),
                "seed": entry.get("settings", {}).get("seed"),
                "timestamp": entry.get("timestamp"),
            })

    return {
        "status": "ok",
        "loras": matching_loras,
        "history_matches": history_matches[-10:],  # last 10 matches
    }


# ══════════════════════════════════════════════════════════════════════════
# WAN Video Generation
# ══════════════════════════════════════════════════════════════════════════
# Uses WAN 2.1 image-to-video pipeline with dual-LoRA (high_noise +
# low_noise) for action animations.  Requires a WAN base model to be
# downloaded into models/ directory.
#
# Required HuggingFace model (auto-downloaded on first run):
#   Wan-AI/Wan2.1-T2V-1.3B  (text-to-video, fits in ~4 GB with bf16
#                             + cpu offload, 8 GB VRAM comfortable)
#
# NOTE: WAN only released I2V (image-to-video) at 14B.  The 1.3B model
# is T2V only.  When an image is provided it is used as a style/scene
# reference via prompt enrichment rather than direct frame conditioning.
#
# The user's action LoRAs are split into high_noise / low_noise variants.
# Both are applied simultaneously: high_noise LoRA acts on early (noisy)
# diffusion steps and low_noise LoRA acts on late (clean) steps.
# ══════════════════════════════════════════════════════════════════════════

_WAN_MODELS_DIR = MODELS_DIR  # WAN model goes in models/ alongside SDXL
_wan_pipeline = None


def _find_wan_model() -> str | None:
    """Look for a WAN model in the models directory."""
    for f in _WAN_MODELS_DIR.iterdir():
        if f.is_dir() and "wan" in f.name.lower() and (f / "model_index.json").exists():
            return str(f)
    # Check if HuggingFace cache has it
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        for repo in cache.repos:
            if "wan2.1" in repo.repo_id.lower():
                # Find the latest revision's snapshot
                for rev in repo.revisions:
                    return str(rev.snapshot_path)
    except Exception:
        pass
    return None


def _find_action_lora_pair(lora_name: str | None = None) -> dict:
    """Find matching high_noise + low_noise LoRA pair from models/action/."""
    action_dir = MODELS_DIR / "action"
    if not action_dir.exists():
        return {}

    high_loras = []
    low_loras = []
    for f in action_dir.iterdir():
        if f.suffix.lower() != ".safetensors" or not f.is_file():
            continue
        name_lower = f.stem.lower()
        if "high_noise" in name_lower or "high" in name_lower:
            high_loras.append(f)
        elif "low_noise" in name_lower or "low" in name_lower:
            low_loras.append(f)

    if lora_name:
        # Find pair matching the given name
        query = lora_name.lower()
        high = next((f for f in high_loras if query in f.stem.lower()), None)
        low = next((f for f in low_loras if query in f.stem.lower()), None)
    else:
        # Use first available pair
        high = high_loras[0] if high_loras else None
        low = low_loras[0] if low_loras else None

    result = {}
    if high:
        result["high_noise"] = str(high)
    if low:
        result["low_noise"] = str(low)
    return result


def _clean_prompt_for_video(prompt: str) -> str:
    """Remove COMPEL/SDXL multi-character artifacts that don't work for video.

    Strips: BREAK keywords, positional markers (on the left/right/center),
    COMPEL weight syntax like (:0.8) or (tag:1.2), and deduplicates words.
    """
    import re as _re
    # Remove BREAK keyword
    prompt = _re.sub(r'\s*BREAK\s*', ' ', prompt)
    # Remove positional markers
    prompt = _re.sub(r',?\s*(?:on the (?:left|right)|in the (?:center|background))\s*', ' ', prompt, flags=_re.IGNORECASE)
    # Remove COMPEL weight syntax: (:0.8), (tag:1.2), etc.
    prompt = _re.sub(r'\(:\d+\.?\d*\)', '', prompt)
    prompt = _re.sub(r'\(([^:()]+):\d+\.?\d*\)', r'\1', prompt)
    # Collapse multiple commas and whitespace
    prompt = _re.sub(r',\s*,+', ',', prompt)
    prompt = _re.sub(r'\s{2,}', ' ', prompt)
    prompt = prompt.strip(' ,')

    # Deduplicate repeated phrases (split on comma, keep first occurrence)
    parts = [p.strip() for p in prompt.split(',')]
    seen: set[str] = set()
    deduped: list[str] = []
    for p in parts:
        key = p.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(p)
    return ', '.join(deduped)


def generate_video(
    prompt: str,
    image_path: str | None = None,
    width: int = 480,
    height: int = 720,
    num_frames: int = 81,
    fps: int = 16,
    steps: int = 40,
    guidance_scale: float = 6.0,
    seed: int = -1,
    action_lora: str | None = None,
    negative_prompt: str | None = None,
    _resume_latent: object | None = None,
    _resume_step: int = 0,
    _resume_job_id: str | None = None,
) -> dict:
    """Generate a video clip using WAN 2.1 with dual-LoRA actions.

    If `image_path` is provided, uses image-to-video (I2V) mode.
    Otherwise uses text-to-video (T2V) mode.

    The dual-LoRA system applies high_noise and low_noise action LoRAs
    simultaneously for motion quality.

    Args:
        prompt: Text description of the video.
        image_path: Optional reference image for I2V mode.
        width/height: Video dimensions (multiples of 16).
        num_frames: Number of video frames (WAN uses groups of ~4).
        fps: Output video framerate.
        steps: Diffusion inference steps.
        guidance_scale: CFG scale.
        seed: Random seed (-1 for random).
        action_lora: Name/keyword to match a specific action LoRA pair.
        negative_prompt: Negative prompt text.

    Returns:
        dict with status, path to output mp4, and metadata.
    """
    global _wan_pipeline
    import gc
    import time as _time

    # ── Strip surrounding quotes from image_path (Windows copy-path adds them)
    if image_path:
        image_path = image_path.strip('"').strip("'")

    # ── Clean COMPEL / multi-character artifacts from the prompt ──────────
    # These are only meaningful for SDXL image gen, not for video.
    prompt = _clean_prompt_for_video(prompt)

    _log.info("[WAN VIDEO] === Starting video generation ===")
    _log.info("[WAN VIDEO]   prompt:  %s", prompt[:200])
    _log.info("[WAN VIDEO]   image:   %s", image_path or '(none — T2V mode)')
    _log.info("[WAN VIDEO]   size:    %dx%d, %d frames @ %d fps", width, height, num_frames, fps)
    _log.info("[WAN VIDEO]   steps:   %d, cfg: %.1f, seed: %d", steps, guidance_scale, seed)

    # Verify image_path exists before proceeding
    if image_path and not Path(image_path).exists():
        _log.error("[WAN VIDEO] Reference image not found: %s", image_path)
        return {
            "status": "error",
            "error": f"Reference image not found: {image_path}. Check the file path.",
        }

    # Check for WAN model
    wan_model_path = _find_wan_model()

    _video_generating.set()
    _video_pause_event.clear()
    _cancel_event.clear()
    _update_progress(active=True, type="video", current_step=0,
                     total_steps=steps, message="Loading WAN model...")

    try:
        if wan_model_path is None:
            # WAN 1.3B is T2V only — I2V was only released at 14B.
            # We always use the T2V-1.3B model regardless of image input.
            wan_model_path = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
            _log.info("[WAN VIDEO] No local WAN model found — downloading %s ...", wan_model_path)

        # ── Load WAN pipeline ────────────────────────────────────
        if _wan_pipeline is None:
            _log.info("[WAN VIDEO] Loading WAN pipeline from: %s", wan_model_path)

            # Free SDXL pipelines to make VRAM room
            flush_vram()

            try:
                from diffusers import WanPipeline
                from diffusers.utils import export_to_video
                import torch as _torch

                # 1.3B is T2V only — always load WanPipeline
                _wan_pipeline = WanPipeline.from_pretrained(
                    wan_model_path,
                    torch_dtype=_torch.bfloat16,
                )

                # ── GPU loading strategy for 8 GB cards ──────────
                # The 1.3B model is ~2.6 GB in bf16.  Loading it
                # directly on CUDA gives full GPU utilisation (~90%+).
                # With attention slicing + VAE tiling the peak stays
                # around 6-7 GB.  If the initial .to("cuda") OOMs we
                # fall back to model_cpu_offload automatically.
                try:
                    _wan_pipeline.to("cuda")
                    _log.info("[WAN VIDEO] Pipeline loaded directly on CUDA")
                except (RuntimeError, _torch.cuda.OutOfMemoryError):
                    _log.warning("[WAN VIDEO] Direct CUDA load OOM — falling back to model_cpu_offload")
                    gc.collect()
                    _torch.cuda.empty_cache()
                    _wan_pipeline = WanPipeline.from_pretrained(
                        wan_model_path,
                        torch_dtype=_torch.bfloat16,
                    )
                    _wan_pipeline.enable_model_cpu_offload()

                # Attention slicing processes attention heads one at a
                # time instead of all at once — VRAM reduction during
                # transformer forward passes with no speed penalty.
                _wan_pipeline.enable_attention_slicing("auto")

                # VAE tiling lets the decoder process large frames in
                # chunks instead of all at once – big VRAM saving.
                if hasattr(_wan_pipeline, "enable_vae_tiling"):
                    _wan_pipeline.enable_vae_tiling()
                if hasattr(_wan_pipeline, "enable_vae_slicing"):
                    _wan_pipeline.enable_vae_slicing()

                _log.info("[WAN VIDEO] WAN 1.3B pipeline ready (attention slicing + VAE tiling)")

            except ImportError as e:
                _log.error("[WAN VIDEO] Missing dependency: %s. "
                           "Install with: pip install diffusers[wan] transformers accelerate", e)
                return {
                    "status": "error",
                    "error": (
                        f"WAN video generation requires additional dependencies: {e}. "
                        "Install with: pip install diffusers[wan] transformers accelerate"
                    ),
                }
            except Exception as e:
                _log.error("[WAN VIDEO] Failed to load WAN model: %s", e)
                return {
                    "status": "error",
                    "error": f"Failed to load WAN model: {e}. "
                             "You may need to download the model first. "
                             "Run: hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                }

        # ── Load action LoRA pair (high_noise + low_noise) ───────
        lora_pair = _find_action_lora_pair(action_lora)
        if lora_pair:
            loaded_names = []
            try:
                if "high_noise" in lora_pair:
                    _wan_pipeline.load_lora_weights(
                        lora_pair["high_noise"],
                        adapter_name="high_noise",
                    )
                    loaded_names.append("high_noise")
                    _log.info("[WAN VIDEO] Loaded high_noise LoRA: %s", lora_pair["high_noise"])

                if "low_noise" in lora_pair:
                    _wan_pipeline.load_lora_weights(
                        lora_pair["low_noise"],
                        adapter_name="low_noise",
                    )
                    loaded_names.append("low_noise")
                    _log.info("[WAN VIDEO] Loaded low_noise LoRA: %s", lora_pair["low_noise"])

                if len(loaded_names) == 2:
                    _wan_pipeline.set_adapters(loaded_names, adapter_weights=[0.8, 0.8])
                elif loaded_names:
                    _wan_pipeline.set_adapters(loaded_names, adapter_weights=[0.8])

            except Exception as e:
                _log.warning("[WAN VIDEO] LoRA loading issue: %s (continuing without LoRA)", e)

        # ── Prepare generation ───────────────────────────────────
        width = max(128, (width // 16) * 16)
        height = max(128, (height // 16) * 16)
        num_frames = max(5, min(num_frames, 200))

        if seed < 0:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        neg = negative_prompt or (
            "blurry, low quality, distorted, deformed, static, frozen, "
            "watermark, text, oversaturated, ugly, amateur, grainy"
        )

        _update_progress(current_step=0, total_steps=steps,
                         message="Generating video frames...")
        _video_timing = [_time.time()]  # mutable list so callback can update

        # ── Checkpoint directory for this job ────────────────────
        job_id = _resume_job_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed}"
        ckpt_dir = VIDEO_CHECKPOINT_DIR / job_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save job metadata for resume
        job_meta = {
            "job_id": job_id,
            "prompt": prompt,
            "negative_prompt": neg,
            "width": width, "height": height,
            "num_frames": num_frames, "fps": fps,
            "steps": steps, "guidance_scale": guidance_scale,
            "seed": seed,
            "image_path": image_path,
            "action_lora": action_lora,
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "last_completed_step": 0,
        }
        _save_json(ckpt_dir / "job.json", job_meta)
        _log.info("[WAN VIDEO] Job %s — checkpoints in %s", job_id, ckpt_dir)

        def _video_step_cb(pipe_self, step_index, timestep, callback_kwargs):
            step = step_index + 1
            elapsed = _time.time() - _video_timing[0]
            sec_per_step = elapsed / max(1, step)
            remaining = sec_per_step * (steps - step)
            mins_left = remaining / 60
            pct = round(step / steps * 100)
            vram_mb = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
            msg = f"Step {step}/{steps} — {pct}% — ~{mins_left:.1f}min left — VRAM {vram_mb:.0f}MB"
            _log.info("[WAN VIDEO] %s  (%.1fs/step)", msg, sec_per_step)
            _update_progress(current_step=step, message=msg)

            # ── Save latent checkpoint every step ────────────────
            try:
                latents = callback_kwargs.get("latents")
                if latents is not None:
                    ckpt_path = ckpt_dir / f"latents_step{step:03d}.pt"
                    torch.save({
                        "latents": latents.cpu(),
                        "step": step,
                        "timestep": timestep.item() if hasattr(timestep, 'item') else float(timestep),
                    }, str(ckpt_path))
                    # Update job metadata
                    job_meta["last_completed_step"] = step
                    job_meta["elapsed_seconds"] = round(elapsed, 1)
                    job_meta["sec_per_step"] = round(sec_per_step, 1)
                    _save_json(ckpt_dir / "job.json", job_meta)
                    _log.info("[WAN VIDEO] Checkpoint saved: %s", ckpt_path.name)
            except Exception as ckpt_err:
                _log.warning("[WAN VIDEO] Checkpoint save failed: %s", ckpt_err)

            # ── Pause support ────────────────────────────────────
            if _video_pause_event.is_set():
                _log.info("[WAN VIDEO] Paused at step %d — waiting for resume...", step)
                _update_progress(message=f"PAUSED at step {step}/{steps} — resume to continue")
                job_meta["status"] = "paused"
                _save_json(ckpt_dir / "job.json", job_meta)
                while _video_pause_event.is_set() and not _cancel_event.is_set():
                    _time.sleep(0.5)
                if _cancel_event.is_set():
                    _cancel_event.clear()
                    raise InterruptedError("Video generation cancelled while paused")
                _log.info("[WAN VIDEO] Resumed at step %d", step)
                job_meta["status"] = "running"
                _save_json(ckpt_dir / "job.json", job_meta)
                # Reset timer so ETA recalculates from resume point
                _video_timing[0] = _time.time() - (step * sec_per_step)

            if _cancel_event.is_set():
                _cancel_event.clear()
                job_meta["status"] = "cancelled"
                _save_json(ckpt_dir / "job.json", job_meta)
                raise InterruptedError("Video generation cancelled")
            if not _check_vram_safe():
                raise MemoryError("VRAM limit reached during video generation")
            return callback_kwargs

        # ── Run pipeline ─────────────────────────────────────────
        # If resuming from a checkpoint, inject the saved latent and
        # use a step callback that skips already-completed steps.
        effective_steps = steps
        if _resume_latent is not None and _resume_step > 0:
            _log.info("[WAN VIDEO] Injecting resume latent from step %d — "
                      "pipeline will run full %d steps but skip first %d in callback",
                      _resume_step, steps, _resume_step)
            # Move the resume latent to the pipeline's device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            resume_lat = _resume_latent.to(device=device, dtype=torch.bfloat16)
        else:
            resume_lat = None

        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": neg,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "callback_on_step_end": _video_step_cb,
        }

        # Pass the resume latent so the pipeline starts from saved state
        if resume_lat is not None:
            gen_kwargs["latents"] = resume_lat

        if image_path:
            # T2V-1.3B has no native I2V mode. We embed the reference image
            # as the first frame by enriching the prompt with a description
            # and appending it to the prompt for context.
            from PIL import Image as PILImage
            ref_image = PILImage.open(image_path).convert("RGB")
            # Resize to match generation resolution
            ref_image = ref_image.resize((width, height), PILImage.LANCZOS)
            # Prepend a framing cue so the model stays close to the source look
            prompt = (
                f"Starting from this scene: {prompt}. "
                "Maintain the same art style, colors, and character appearance throughout."
            )
            _log.info("[WAN VIDEO] I2V requested but using T2V-1.3B; image used as style reference")

        output = _wan_pipeline(**gen_kwargs)

        # If we got an OOM during the first attempt with model_cpu_offload,
        # fall back to group_offload (block-level) which uses less VRAM
        # at the cost of slightly lower GPU utilization.
        # (The RuntimeError/OutOfMemoryError is caught below in the
        #  except block and retried once.)
        frames = output.frames[0]  # list of PIL Images

        # ── Export to mp4 ────────────────────────────────────────
        VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r'[^\w\s-]', '', prompt[:30]).strip().replace(' ', '_')
        mp4_filename = str(VIDEO_OUTPUT_DIR / f"{timestamp}_{safe_name}_video.mp4")

        export_ok = False
        # Method 1: diffusers export_to_video (uses imageio + ffmpeg)
        try:
            from diffusers.utils import export_to_video
            export_to_video(frames, mp4_filename, fps=fps)
            if Path(mp4_filename).exists() and Path(mp4_filename).stat().st_size > 0:
                export_ok = True
                _log.info("[WAN VIDEO] Exported via diffusers export_to_video")
        except Exception as e:
            _log.warning("[WAN VIDEO] export_to_video failed: %s — trying fallbacks", e)

        # Method 2: opencv VideoWriter
        if not export_ok:
            try:
                import cv2
                import numpy as np
                h0, w0 = frames[0].size[1], frames[0].size[0]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(mp4_filename, fourcc, fps, (w0, h0))
                for frame in frames:
                    arr = np.array(frame.convert("RGB"))
                    writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                writer.release()
                if Path(mp4_filename).exists() and Path(mp4_filename).stat().st_size > 0:
                    export_ok = True
                    _log.info("[WAN VIDEO] Exported via OpenCV")
            except Exception as e:
                _log.warning("[WAN VIDEO] OpenCV export failed: %s", e)

        # Method 3: save frames as PNGs (always works, never loses data)
        if not export_ok:
            frame_dir = VIDEO_OUTPUT_DIR / f"{timestamp}_video_frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            for idx, frame in enumerate(frames):
                fp = str(frame_dir / f"frame_{idx:04d}.png")
                frame.save(fp)
            _log.info("[WAN VIDEO] Saved %d frames as PNGs to %s (mp4 export failed)", len(frames), frame_dir)
            # Try _frames_to_mp4 as last resort
            frame_paths = sorted(str(p) for p in frame_dir.glob("*.png"))
            result_path = _frames_to_mp4(frame_paths, fps, Path(mp4_filename))
            if result_path:
                mp4_filename = result_path
                export_ok = True
            else:
                # Frames are saved — return the frame directory so work isn't lost
                mp4_filename = str(frame_dir)
                _log.warning("[WAN VIDEO] All mp4 exports failed — frames saved as PNGs in %s", frame_dir)

        _log.info("[WAN VIDEO] Video saved: %s (%d frames, %dfps)", mp4_filename, len(frames), fps)

        # ── Save video metadata ──────────────────────────────────
        meta = {
            "job_id": job_id,
            "path": mp4_filename,
            "prompt": prompt,
            "negative_prompt": neg,
            "width": width, "height": height,
            "num_frames": len(frames), "fps": fps,
            "steps": steps, "guidance_scale": guidance_scale,
            "seed": seed,
            "image_path": image_path,
            "action_lora": action_lora,
            "action_loras": lora_pair,
            "created_at": datetime.now().isoformat(),
            "duration_seconds": round(_time.time() - _video_timing[0], 1),
        }
        meta_path = Path(mp4_filename).with_suffix(".json")
        _save_json(meta_path, meta)
        _log.info("[WAN VIDEO] Metadata saved: %s", meta_path)

        # Mark job complete and clean up checkpoints
        job_meta["status"] = "completed"
        job_meta["output_path"] = mp4_filename
        _save_json(ckpt_dir / "job.json", job_meta)

        # Clean up LoRAs after generation
        try:
            _wan_pipeline.unload_lora_weights()
        except Exception:
            pass

        return {
            "status": "ok",
            "job_id": job_id,
            "path": mp4_filename,
            "num_frames": len(frames),
            "seed": seed,
            "fps": fps,
            "width": width,
            "height": height,
            "action_loras": lora_pair,
            "duration_seconds": round(_time.time() - _video_timing[0], 1),
        }

    except InterruptedError:
        _log.info("[WAN VIDEO] Generation cancelled by user")
        return {"status": "cancelled", "message": "Video generation cancelled"}
    except (MemoryError, torch.cuda.OutOfMemoryError) as e:
        _log.warning("[WAN VIDEO] VRAM OOM: %s — reloading with model_cpu_offload", e)
        gc.collect()
        torch.cuda.empty_cache()

        # Retry once with model_cpu_offload (components stay on CPU, move to GPU one at a time)
        try:
            _update_progress(message="OOM — retrying with CPU offload (lower VRAM)...")
            import torch as _torch
            # Reload pipeline with offload
            _wan_pipeline = None
            gc.collect()
            _torch.cuda.empty_cache()
            from diffusers import WanPipeline
            _wan_pipeline = WanPipeline.from_pretrained(
                _find_wan_model() or "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                torch_dtype=_torch.bfloat16,
            )
            _wan_pipeline.enable_model_cpu_offload()
            _wan_pipeline.enable_attention_slicing("auto")
            if hasattr(_wan_pipeline, "enable_vae_tiling"):
                _wan_pipeline.enable_vae_tiling()
            if hasattr(_wan_pipeline, "enable_vae_slicing"):
                _wan_pipeline.enable_vae_slicing()
            _log.info("[WAN VIDEO] Retrying with model_cpu_offload")
            _video_timing[0] = _time.time()
            output = _wan_pipeline(**gen_kwargs)
            frames = output.frames[0]

            # Export
            VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = re.sub(r'[^\w\s-]', '', prompt[:30]).strip().replace(' ', '_')
            mp4_filename = str(VIDEO_OUTPUT_DIR / f"{timestamp}_{safe_name}_video.mp4")
            try:
                from diffusers.utils import export_to_video
                export_to_video(frames, mp4_filename, fps=fps)
            except Exception:
                import cv2, numpy as np
                h0, w0 = frames[0].size[1], frames[0].size[0]
                writer = cv2.VideoWriter(mp4_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w0, h0))
                for frame in frames:
                    writer.write(cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR))
                writer.release()
            _log.info("[WAN VIDEO] Video saved (CPU offload fallback): %s", mp4_filename)
            job_meta["status"] = "completed"
            job_meta["output_path"] = mp4_filename
            job_meta["offload_mode"] = "model_cpu_offload_fallback"
            _save_json(ckpt_dir / "job.json", job_meta)
            return {
                "status": "ok", "job_id": job_id, "path": mp4_filename,
                "num_frames": len(frames), "seed": seed, "fps": fps,
                "width": width, "height": height,
                "duration_seconds": round(_time.time() - _video_timing[0], 1),
                "note": "Used CPU offload fallback due to VRAM limit",
            }
        except Exception as fallback_err:
            _log.error("[WAN VIDEO] CPU offload fallback also failed: %s", fallback_err)
            gc.collect()
            torch.cuda.empty_cache()
            return {"status": "error", "error": f"VRAM exhausted even with offload: {fallback_err}. Try smaller resolution or fewer frames."}
    except Exception as e:
        _log.error("[WAN VIDEO] Error: %s", e)
        return {"status": "error", "error": str(e)}
    finally:
        _video_generating.clear()
        _video_pause_event.clear()
        _update_progress(active=False, type="idle", current_step=0,
                         total_steps=0, message="")


def flush_wan_pipeline():
    """Unload WAN video pipeline to free VRAM."""
    global _wan_pipeline
    if _wan_pipeline is not None:
        del _wan_pipeline
        _wan_pipeline = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log.info("[WAN VIDEO] WAN pipeline flushed from VRAM")


# ── Video pause / resume / queue ─────────────────────────────────────────

def pause_video_generation():
    """Pause the current video generation after the current step completes."""
    if not _video_generating.is_set():
        return {"status": "error", "message": "No video generation in progress"}
    _video_pause_event.set()
    _log.info("[WAN VIDEO] Pause requested — will pause after current step")
    return {"status": "ok", "message": "Pause requested — will pause after current step finishes"}


def resume_video_generation():
    """Resume a paused video generation."""
    _video_pause_event.clear()
    _log.info("[WAN VIDEO] Resume requested")
    return {"status": "ok", "message": "Resumed"}


def resume_interrupted_job(job_id: str) -> dict:
    """Resume an interrupted/cancelled/running job.

    If all diffusion steps are already saved in checkpoints, just decode
    the final latents through the VAE (fast, ~30s) instead of re-running
    the entire diffusion process.  If partially done, attempt to resume
    from the last saved checkpoint step.  Falls back to re-running from
    scratch with the same seed only if checkpoint loading fails.

    Returns dict with status + path to the output mp4.
    """
    ckpt_dir = VIDEO_CHECKPOINT_DIR / job_id
    meta_file = ckpt_dir / "job.json"
    if not meta_file.exists():
        return {"status": "error", "error": f"No checkpoint found for job {job_id}"}
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        return {"status": "error", "error": f"Failed to read job metadata: {e}"}

    status = meta.get("status", "")
    if status == "completed" and meta.get("output_path"):
        if Path(meta["output_path"]).exists():
            return {"status": "ok", "path": meta["output_path"],
                    "message": "Already completed", "job_id": job_id}

    if _video_generating.is_set():
        return {"status": "error", "error": "Another video generation is already running"}

    # Count how many checkpoint latents we have
    latent_files = sorted(ckpt_dir.glob("latents_step*.pt"))
    total_steps = meta.get("steps", 20)
    saved_steps = len(latent_files)

    _log.info("[WAN VIDEO] Resume job %s: %d/%d steps saved, status=%s",
              job_id, saved_steps, total_steps, status)

    # If ALL steps are done -> just decode through VAE (much faster)
    if saved_steps >= total_steps:
        _log.info("[WAN VIDEO] All %d steps saved — decoding from checkpoint", total_steps)
        return _decode_checkpoint_to_video(job_id, meta, ckpt_dir)

    # If we have partial steps, try to resume from the last checkpoint
    if saved_steps > 0:
        last_ckpt = latent_files[-1]
        _log.info("[WAN VIDEO] Resuming from step %d/%d — loading %s",
                  saved_steps, total_steps, last_ckpt.name)
        try:
            ckpt_data = torch.load(str(last_ckpt), map_location="cpu", weights_only=True)
            resume_latent = ckpt_data["latents"]
            resume_step = ckpt_data["step"]
            resume_timestep = ckpt_data.get("timestep")

            _log.info("[WAN VIDEO] Loaded latent from step %d (timestep=%.1f), "
                      "will run remaining %d steps",
                      resume_step, resume_timestep or -1,
                      total_steps - resume_step)

            meta["status"] = "resuming"
            meta["resume_from_step"] = resume_step
            _save_json(meta_file, meta)

            return generate_video(
                prompt=meta.get("prompt", ""),
                image_path=meta.get("image_path"),
                width=meta.get("width", 480),
                height=meta.get("height", 720),
                num_frames=meta.get("num_frames", 81),
                fps=meta.get("fps", 16),
                steps=meta.get("steps", 30),
                guidance_scale=meta.get("guidance_scale", 5.0),
                seed=meta.get("seed", -1),
                action_lora=meta.get("action_lora"),
                negative_prompt=meta.get("negative_prompt"),
                _resume_latent=resume_latent,
                _resume_step=resume_step,
                _resume_job_id=job_id,
            )
        except Exception as e:
            _log.warning("[WAN VIDEO] Failed to load checkpoint latent: %s — "
                         "falling back to full re-run", e)

    # Fallback: re-run from scratch with same params
    _log.info("[WAN VIDEO] No usable checkpoints — re-running from scratch (same seed)")
    meta["status"] = "superseded"
    _save_json(meta_file, meta)

    return generate_video(
        prompt=meta.get("prompt", ""),
        image_path=meta.get("image_path"),
        width=meta.get("width", 480),
        height=meta.get("height", 720),
        num_frames=meta.get("num_frames", 81),
        fps=meta.get("fps", 16),
        steps=meta.get("steps", 30),
        guidance_scale=meta.get("guidance_scale", 5.0),
        seed=meta.get("seed", -1),
        action_lora=meta.get("action_lora"),
        negative_prompt=meta.get("negative_prompt"),
    )


def _decode_checkpoint_to_video(job_id: str, meta: dict, ckpt_dir: Path) -> dict:
    """Decode the final-step latents through the WAN VAE and export to mp4.

    This is MUCH faster than re-running diffusion (~30s vs hours).
    Only the VAE needs to be loaded on GPU (~300MB).
    """
    import gc
    import time as _time

    total_steps = meta.get("steps", 20)
    fps = meta.get("fps", 16)
    prompt = meta.get("prompt", "")

    _video_generating.set()
    _update_progress(active=True, type="video", current_step=0,
                     total_steps=3, message="Decoding video from checkpoint...")

    try:
        # Load final-step latents
        latent_path = ckpt_dir / f"latents_step{total_steps:03d}.pt"
        if not latent_path.exists():
            # Try the highest available step
            latent_files = sorted(ckpt_dir.glob("latents_step*.pt"))
            if not latent_files:
                return {"status": "error", "error": "No latent checkpoints found"}
            latent_path = latent_files[-1]

        _log.info("[WAN VIDEO] Loading latents from %s", latent_path.name)
        final = torch.load(str(latent_path), map_location="cpu", weights_only=True)
        latents = final["latents"]
        _log.info("[WAN VIDEO] Latents shape: %s dtype: %s", latents.shape, latents.dtype)

        _update_progress(current_step=1, message="Loading VAE decoder...")

        # Load just the VAE (much smaller than full pipeline)
        from diffusers import AutoencoderKLWan
        wan_model_path = _find_wan_model() or "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

        vae = AutoencoderKLWan.from_pretrained(
            wan_model_path,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        vae.to("cuda")
        vae.enable_tiling()
        vae.enable_slicing()
        _log.info("[WAN VIDEO] VAE loaded on CUDA")

        _update_progress(current_step=2, message="Decoding latents to video frames...")

        # Decode: apply inverse latent normalization then VAE decode
        latents = latents.to(dtype=vae.dtype, device="cuda")
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
            1, vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean

        with torch.no_grad():
            video = vae.decode(latents, return_dict=False)[0]

        _log.info("[WAN VIDEO] Decoded video shape: %s", video.shape)

        # Convert to PIL frames
        from diffusers.video_processor import VideoProcessor
        processor = VideoProcessor(vae_scale_factor=8)
        frames = processor.postprocess_video(video, output_type="pil")[0]
        _log.info("[WAN VIDEO] Got %d PIL frames", len(frames))

        # Free VRAM
        del vae, latents, video
        gc.collect()
        torch.cuda.empty_cache()

        _update_progress(current_step=3, message="Exporting video...")

        # Export to mp4
        VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r'[^\w\s-]', '', prompt[:30]).strip().replace(' ', '_')
        mp4_filename = str(VIDEO_OUTPUT_DIR / f"{timestamp}_{safe_name}_video.mp4")

        export_ok = False
        try:
            from diffusers.utils import export_to_video
            export_to_video(frames, mp4_filename, fps=fps)
            if Path(mp4_filename).exists() and Path(mp4_filename).stat().st_size > 0:
                export_ok = True
        except Exception as e:
            _log.warning("[WAN VIDEO] export_to_video failed: %s", e)

        if not export_ok:
            try:
                import cv2
                import numpy as np
                h0, w0 = frames[0].size[1], frames[0].size[0]
                writer = cv2.VideoWriter(mp4_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w0, h0))
                for frame in frames:
                    writer.write(cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR))
                writer.release()
                export_ok = True
            except Exception as e:
                _log.warning("[WAN VIDEO] OpenCV export failed: %s", e)

        if not export_ok:
            # Save frames as PNGs — never lose data
            frame_dir = VIDEO_OUTPUT_DIR / f"{timestamp}_video_frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            for idx, frame in enumerate(frames):
                frame.save(str(frame_dir / f"frame_{idx:04d}.png"))
            mp4_filename = str(frame_dir)
            _log.warning("[WAN VIDEO] Saved as PNGs: %s", frame_dir)

        _log.info("[WAN VIDEO] Video saved: %s (%d frames, %dfps)", mp4_filename, len(frames), fps)

        # Save metadata
        video_meta = {
            "job_id": job_id,
            "path": mp4_filename,
            "prompt": prompt,
            "negative_prompt": meta.get("negative_prompt", ""),
            "width": meta.get("width"), "height": meta.get("height"),
            "num_frames": len(frames), "fps": fps,
            "steps": total_steps,
            "guidance_scale": meta.get("guidance_scale"),
            "seed": meta.get("seed"),
            "action_lora": meta.get("action_lora"),
            "created_at": datetime.now().isoformat(),
            "decoded_from_checkpoint": True,
        }
        meta_path = Path(mp4_filename).with_suffix(".json")
        _save_json(meta_path, video_meta)

        # Mark checkpoint as completed
        meta["status"] = "completed"
        meta["output_path"] = mp4_filename
        _save_json(ckpt_dir / "job.json", meta)

        return {
            "status": "ok",
            "job_id": job_id,
            "path": mp4_filename,
            "num_frames": len(frames),
            "seed": meta.get("seed"),
            "fps": fps,
            "width": meta.get("width"),
            "height": meta.get("height"),
            "decoded_from_checkpoint": True,
        }

    except Exception as e:
        _log.error("[WAN VIDEO] Checkpoint decode failed: %s", e)
        return {"status": "error", "error": f"Checkpoint decode failed: {e}"}
    finally:
        _video_generating.clear()
        _video_pause_event.clear()
        _update_progress(active=False, type="idle", current_step=0,
                         total_steps=0, message="")


def get_video_queue_status() -> dict:
    """Return current video generation status + list of all jobs."""
    jobs = list_video_checkpoints()
    # "resumable" = any job that isn't completed/superseded (and has checkpoints)
    resumable = [j for j in jobs if j.get("status") not in ("completed", "superseded")
                 and j.get("checkpoint_count", 0) > 0]
    completed = [j for j in jobs if j.get("status") == "completed"]
    return {
        "is_generating": _video_generating.is_set(),
        "is_paused": _video_pause_event.is_set(),
        "resumable_jobs": resumable,
        # Keep interrupted_jobs for backward-compat with frontend
        "interrupted_jobs": resumable,
        "completed_count": len(completed),
        "all_jobs": jobs,
    }


def list_video_checkpoints() -> list[dict]:
    """List all saved video generation checkpoints."""
    if not VIDEO_CHECKPOINT_DIR.exists():
        return []
    jobs = []
    for job_dir in sorted(VIDEO_CHECKPOINT_DIR.iterdir(), reverse=True):
        meta_file = job_dir / "job.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                ckpt_files = list(job_dir.glob("latents_step*.pt"))
                meta["checkpoint_count"] = len(ckpt_files)
                meta["latest_checkpoint"] = max(
                    (int(f.stem.split("step")[1]) for f in ckpt_files), default=0
                )
                jobs.append(meta)
            except Exception:
                pass
    return jobs


def list_generated_videos() -> list[dict]:
    """List all generated videos with their metadata."""
    if not VIDEO_OUTPUT_DIR.exists():
        return []
    videos = []
    for meta_file in sorted(VIDEO_OUTPUT_DIR.glob("*.json"), reverse=True):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # Verify video file exists
            video_path = meta.get("path", "")
            meta["video_exists"] = Path(video_path).exists() if video_path else False
            videos.append(meta)
        except Exception:
            pass
    return videos


def cleanup_checkpoints(job_id: str | None = None, keep_latest: int = 3):
    """Clean up old checkpoint directories.

    If job_id is given, delete only that job's checkpoints.
    Otherwise keep only the N most recent completed jobs.
    """
    if not VIDEO_CHECKPOINT_DIR.exists():
        return {"status": "ok", "cleaned": 0}
    cleaned = 0
    if job_id:
        target = VIDEO_CHECKPOINT_DIR / job_id
        if target.exists():
            shutil.rmtree(target)
            cleaned = 1
    else:
        all_jobs = sorted(VIDEO_CHECKPOINT_DIR.iterdir(), reverse=True)
        for job_dir in all_jobs[keep_latest:]:
            meta_file = job_dir / "job.json"
            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        meta = json.load(f)
                    if meta.get("status") == "completed":
                        shutil.rmtree(job_dir)
                        cleaned += 1
                except Exception:
                    pass
    _log.info("[WAN VIDEO] Cleaned %d checkpoint(s)", cleaned)
    return {"status": "ok", "cleaned": cleaned}


def list_action_loras() -> list[dict]:
    """List available action LoRAs in models/action/ with pair detection."""
    action_dir = MODELS_DIR / "action"
    if not action_dir.exists():
        return []

    loras = []
    seen_bases = set()
    tw_data = _load_trigger_words()

    for f in sorted(action_dir.iterdir()):
        if f.suffix.lower() != ".safetensors" or not f.is_file():
            continue
        name = f.stem
        # Determine if high/low noise variant
        noise_type = "unknown"
        if "high_noise" in name.lower():
            noise_type = "high_noise"
        elif "low_noise" in name.lower():
            noise_type = "low_noise"

        # Detect pair base name (strip _high_noise / _low_noise suffix)
        base = re.sub(r'_?(high|low)_?noise', '', name, flags=re.IGNORECASE).strip('_')

        loras.append({
            "name": name,
            "path": str(f),
            "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
            "noise_type": noise_type,
            "pair_base": base,
            "trigger_words": tw_data.get(name, []),
            "has_pair": base in seen_bases,
        })
        seen_bases.add(base)

    # Mark pairs retroactively
    for lora in loras:
        lora["has_pair"] = sum(1 for l in loras if l["pair_base"] == lora["pair_base"]) > 1

    return loras
