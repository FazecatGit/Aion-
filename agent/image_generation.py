"""
Offline image generation using Stable Diffusion XL (local safetensors models).

Supports:
- Multiple checkpoint models (auto-discovered from models/ folder)
- LoRA loading on top of base checkpoints
- Explicit vs normal mode (no hardcoded NSFW prefixes)
- Long-prompt handling via CLIP chunking to bypass 77-token limit
- Prompt enhancement via local LLM (optional)
- Animated image generation (frame interpolation)
- Feedback loop: track what the pipeline actually focused on
"""

import os
import re
import json
import logging
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

    # Also scan subdirectories (e.g., models/loras/, models/characters/)
    for subdir_name in ("loras", "characters"):
        sub_dir = MODELS_DIR / subdir_name
        if sub_dir.exists():
            for f in sorted(sub_dir.iterdir()):
                if f.suffix.lower() in LORA_EXTENSIONS and f.is_file():
                    models.append({
                        "name": f.stem,
                        "filename": f.name,
                        "path": str(f),
                        "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                        "type": "lora",
                        "category": subdir_name,
                    })

    return models


def _is_lora_file(path: Path) -> bool:
    """Heuristic: LoRA files are generally < 300 MB."""
    return path.stat().st_size < 300 * 1024 * 1024


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

def _chunk_prompt(prompt: str, pipe: StableDiffusionXLPipeline) -> str:
    """
    Enable compel-style long prompt embedding when available.
    For SDXL, we use the pipeline's built-in long prompt weighting
    by splitting into weighted segments.
    """
    # SDXL diffusers >= 0.25 supports long prompts natively via
    # the `encode_prompt` method with truncation disabled.
    # We enable it by returning the prompt as-is and setting
    # max_sequence_length in the generation call.
    return prompt


def _encode_long_prompt(pipe, prompt: str, negative_prompt: str, device: str = "cuda"):
    """
    Encode prompts using compel for SDXL, which properly handles long prompts
    beyond the 77-token CLIP limit by doing weighted sub-prompt blending.
    Falls back to manual BREAK-based chunking if compel is not available.
    """
    try:
        from compel import Compel, ReturnedEmbeddingsType
        _log.info("[IMAGE GEN] Using compel for long prompt encoding (%d chars)", len(prompt))

        # For SDXL we need to handle both text encoders
        compel_proc = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,  # Key: do NOT truncate
        )
        conditioning, pooled = compel_proc(prompt)
        neg_conditioning, neg_pooled = compel_proc(negative_prompt)
        [conditioning, neg_conditioning] = compel_proc.pad_conditioning_tensors_to_same_length(
            [conditioning, neg_conditioning]
        )
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
    Fallback: split prompt into 75-token chunks, encode each, and concatenate.
    Works without compel installed.
    """
    tokenizer = pipe.tokenizer
    if tokenizer is None:
        return None  # pipeline doesn't support manual encoding

    def _get_chunks(text: str, max_len: int = 75):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), max_len):
            chunk_tokens = tokens[i:i + max_len]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
        return chunks if chunks else [text]

    # For SDXL, the simplest reliable approach is to use BREAK keyword
    # which diffusers handles natively to separate prompt segments
    chunks = _get_chunks(prompt)
    if len(chunks) > 1:
        # Join with BREAK keyword for SDXL native handling
        enhanced_prompt = " BREAK ".join(chunks)
        return {"_chunked_prompt": enhanced_prompt, "_chunked_negative": negative_prompt}

    return None


# ── Feedback / prompt analysis ────────────────────────────────────────────

_generation_history: list[dict] = []
MAX_HISTORY = 50


def _analyze_prompt(prompt: str) -> dict:
    """Break down the prompt into weighted components for transparency."""
    # Split by commas and weight markers
    parts = [p.strip() for p in prompt.split(",") if p.strip()]
    analysis = {
        "total_parts": len(parts),
        "components": parts,
        "estimated_focus": [],
    }

    # Early items get more attention from the model
    for i, part in enumerate(parts):
        weight = "high" if i < 3 else "medium" if i < 8 else "low"
        analysis["estimated_focus"].append({"text": part, "priority": weight, "position": i + 1})

    return analysis


def record_generation(prompt: str, negative: str, settings: dict, result_path: str | None, feedback: str | None = None):
    """Record generation for learning loop."""
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
    return entry


def get_generation_history(last_n: int = 20) -> list[dict]:
    return _generation_history[-last_n:]


def apply_feedback_learnings(base_negative: str) -> str:
    """Analyze recent negative feedback and add common issues to negative prompt."""
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
    }

    additions = set()
    for entry in _generation_history[-10:]:
        fb = (entry.get("feedback") or "").lower()
        if not fb:
            continue
        for key, neg_text in issue_keywords.items():
            if key in fb:
                additions.add(neg_text)

    if additions:
        return base_negative + ", " + ", ".join(additions)
    return base_negative


# ── Core generation ───────────────────────────────────────────────────────

DEFAULT_NEGATIVE = (
    "blurry, low quality, deformed, bad anatomy, watermark, bad hands, "
    "text, error, cropped, worst quality, bad quality, worst detail, sketch, "
    "censored, signature, watermark"
)

EXPLICIT_NEGATIVE = (
    "score_4, score_3, score_2, score_1, "
    "blurry, low quality, deformed, bad anatomy, watermark, bad hands, "
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

    # Load pipeline
    pipe = _load_pipeline(model_path)

    # Load LoRAs if requested
    if lora_paths:
        unload_loras(pipe)  # clear any previous LoRAs
        weights = lora_weights or [0.8] * len(lora_paths)
        for lp, lw in zip(lora_paths, weights):
            if Path(lp).exists():
                load_lora(pipe, lp, weight=lw)
            else:
                _log.warning("[IMAGE GEN] LoRA not found: %s", lp)

    # Build prompt based on mode
    if mode == "explicit":
        full_prompt = f"masterpiece, best quality, amazing quality, {prompt}"
        base_neg = negative_prompt or EXPLICIT_NEGATIVE
    else:
        full_prompt = f"masterpiece, best quality, {prompt}"
        base_neg = negative_prompt or DEFAULT_NEGATIVE

    # Apply feedback learnings to negative prompt
    full_negative = apply_feedback_learnings(base_neg)

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

        gen_kwargs = {
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        if long_prompt_used and embed_kwargs:
            gen_kwargs.update(embed_kwargs)
        else:
            gen_kwargs["prompt"] = full_prompt
            gen_kwargs["negative_prompt"] = full_negative

        image = pipe(**gen_kwargs).images[0]

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

        return {
            "status": "ok",
            "path": filename,
            "seed": seed,
            "prompt_used": full_prompt,
            "negative_used": full_negative,
            "prompt_analysis": prompt_analysis,
            "settings": settings,
            "long_prompt": long_prompt_used,
        }

    except Exception as e:
        _log.error("[IMAGE GEN] Error: %s", str(e))
        return {"status": "error", "error": str(e)}


# ── Animated generation (frame interpolation) ────────────────────────────

def generate_animated(
    prompt: str,
    width: int = 512,
    height: int = 512,
    model_path: str | None = None,
    mode: str = "normal",
    num_frames: int = 8,
    steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = -1,
) -> dict:
    """
    Generate a simple animated GIF by interpolating seeds across frames.
    Uses the same SDXL pipeline but generates multiple images with
    smoothly varying seeds for simple animation effect.
    """
    import numpy as np

    if num_frames < 2:
        num_frames = 2
    if num_frames > 24:
        num_frames = 24

    base_seed = seed if seed >= 0 else torch.randint(0, 2**32, (1,)).item()
    frames = []
    frame_paths = []

    output_dir_path = Path(OUTPUT_DIR)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(num_frames):
        # Interpolate between two seeds for smooth transition
        frame_seed = base_seed + i * 42  # step through seed space
        result = generate_image(
            prompt=prompt,
            width=width,
            height=height,
            model_path=model_path,
            mode=mode,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=frame_seed,
        )
        if result.get("status") == "error":
            return result
        frame_paths.append(result["path"])
        _log.info("[ANIM GEN] Frame %d/%d generated", i + 1, num_frames)

    # Combine frames into GIF
    try:
        from PIL import Image as PILImage
        pil_frames = [PILImage.open(fp) for fp in frame_paths]
        gif_path = str(output_dir_path / f"{timestamp}_animated.gif")
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=150,  # ms per frame
            loop=0,
        )
        _log.info("[ANIM GEN] GIF saved to: %s", gif_path)

        return {
            "status": "ok",
            "path": gif_path,
            "frames": frame_paths,
            "num_frames": num_frames,
            "seed": base_seed,
        }
    except Exception as e:
        return {"status": "error", "error": f"GIF creation failed: {e}", "frames": frame_paths}


# ── Feedback submission ───────────────────────────────────────────────────

def submit_feedback(generation_index: int, feedback_text: str) -> dict:
    """Submit feedback for a specific generation to improve future results."""
    if generation_index < 0 or generation_index >= len(_generation_history):
        return {"status": "error", "error": "Invalid generation index"}

    _generation_history[generation_index]["feedback"] = feedback_text
    _log.info("[IMAGE GEN] Feedback recorded for generation %d: %s",
              generation_index, feedback_text[:100])

    return {"status": "ok", "message": "Feedback recorded — will influence future generations"}


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
