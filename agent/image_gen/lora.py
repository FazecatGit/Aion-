"""LoRA management — trigger words, categories, loading/unloading, search."""

import json
import logging
import re
from pathlib import Path

from diffusers import StableDiffusionXLPipeline

from .core import MODELS_DIR, LORA_EXTENSIONS, _active_loras, _log

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
    colon_entries = []
    for e in raw_entries:
        word = e["word"]
        if not word:
            continue
        colon_match = re.match(r'^(.+?):\s*(.+)$', word)
        if colon_match:
            label = colon_match.group(1).strip()
            rest = colon_match.group(2).strip()
            colon_entries.append((label, rest, e))

    if len(colon_entries) >= 2 and len(colon_entries) >= len(raw_entries) * 0.5:
        primary = ""
        outfits = {}

        for label, rest, _orig in colon_entries:
            tags = [{"word": t.strip(), "weight": 1.0} for t in rest.split(",") if t.strip()]
            is_base = "base" in label.lower() or "custom outfit" in label.lower()

            if is_base and not primary:
                primary = tags[0]["word"] if tags else label
                outfits[label] = tags
            else:
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
        primary_fmt = format_trigger_word({"word": parsed["primary"], "weight": 1.0})
        if primary_fmt:
            parts.append(primary_fmt)

        outfit_tags: list[dict] = []
        if selected_outfit and selected_outfit in parsed["outfits"]:
            outfit_tags = parsed["outfits"][selected_outfit]
        elif parsed["outfits"]:
            first_key = next(iter(parsed["outfits"]))
            outfit_tags = parsed["outfits"][first_key]

        for tag in outfit_tags:
            fmt = format_trigger_word(tag)
            if fmt:
                parts.append(fmt)
    else:
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


def load_lora(pipe: StableDiffusionXLPipeline, lora_path: str, weight: float = 0.8):
    """Load a LoRA adapter on top of the current pipeline."""
    import agent.image_gen.core as _core
    from safetensors.torch import load_file as safe_load
    from pathlib import Path

    _log.info("[IMAGE GEN] Loading LoRA: %s (weight=%.2f)", lora_path, weight)

    # ── Quick architecture compatibility check ─────────────────────────
    # Detect SD1.5 LoRAs being loaded on SDXL (or vice-versa) before
    # diffusers throws an opaque state_dict error.
    lp = Path(lora_path)
    if lp.suffix.lower() == ".safetensors":
        try:
            sd = safe_load(str(lp), device="cpu")
            # SD1.5 cross-attn keys project from 768-dim text encoder;
            # SDXL projects from 2048-dim.  Check any attn2.to_k key.
            for key, tensor in sd.items():
                if "attn2.to_k" in key and "lora_A" in key:
                    dim = tensor.shape[-1]  # last dim is always the input
                    if dim == 768:
                        raise ValueError(
                            f"LoRA '{lp.stem}' was trained for SD 1.5 "
                            f"(cross-attn dim=768) but the current model is SDXL "
                            f"(cross-attn dim=2048). This LoRA is incompatible — "
                            f"use an SDXL version instead."
                        )
                    break  # only need to check one key
            del sd  # free memory
        except ValueError:
            raise
        except Exception as e:
            _log.warning("[IMAGE GEN] LoRA pre-check skipped: %s", e)

    try:
        pipe.load_lora_weights(lora_path)
    except RuntimeError as e:
        err_msg = str(e)
        if "size mismatch" in err_msg:
            raise ValueError(
                f"LoRA '{lp.stem}' is incompatible with the current model "
                f"(tensor shape mismatch). This usually means the LoRA was "
                f"trained for a different architecture (e.g. SD1.5 vs SDXL). "
                f"Use a LoRA matching your checkpoint."
            ) from e
        raise
    pipe.fuse_lora(lora_scale=weight)
    _core._active_loras.append(lora_path)


def unload_loras(pipe: StableDiffusionXLPipeline):
    """Remove all LoRA adapters."""
    import agent.image_gen.core as _core
    if _core._active_loras:
        pipe.unfuse_lora()
        pipe.unload_lora_weights()
        _core._active_loras = []
        _log.info("[IMAGE GEN] All LoRAs unloaded")


def search_characters(query: str) -> dict:
    """Search all LoRA categories and past generations matching a query.

    Searches file names, trigger words, and generation history prompts.
    Returns matching LoRA adapters and recent history entries.
    """
    from .history import _generation_history
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

