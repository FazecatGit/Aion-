"""Core image generation — generate_image, regional conditioning, preview, img2img."""

import re
import time
import logging
import threading
from pathlib import Path
from datetime import datetime

import torch
from diffusers import StableDiffusionXLPipeline

from brain.config import OUTPUT_DIR

from .core import (
    MODELS_DIR, _log,
    _loaded_pipelines, _active_model, _active_loras,
    _cancel_event, _update_progress,
)
from .models import (
    list_models, _load_pipeline, evict_ollama_models,
    _resolve_model_path,
)
from .lora import (
    _load_trigger_words, parse_outfit_groups, parse_trigger_word_entry,
    build_trigger_prompt, load_lora, unload_loras,
)
from .styles import _apply_art_style
from .prompts import (
    _analyze_prompt, _encode_long_prompt,
    _structure_multi_angle_prompt, _structure_multi_character_prompt,
    _parse_regional_sections, _build_anti_bleed_negative,
    _encode_single_sdxl, _create_spatial_masks,
)


# ── Negative prompt defaults ─────────────────────────────────────────────

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


# ── Img2Img pipeline cache ───────────────────────────────────────────────

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


# ── Regional conditioning generation ────────────────────────────────────

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
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pipe.unet.dtype

    parsed = _parse_regional_sections(prompt)
    if parsed is None:
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

    # Encode prompts
    neg_embeds, neg_pooled = _encode_single_sdxl(pipe, negative_prompt, device)

    char_embeds_list = []
    for r in regions:
        combined = f"{shared_prompt}, {r['prompt']}"
        embeds, pooled = _encode_single_sdxl(pipe, combined, device)
        char_embeds_list.append((embeds, pooled))

    # Create spatial masks
    masks = _create_spatial_masks(regions, latent_w, latent_h, device, dtype)

    # SDXL time IDs
    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]], dtype=dtype,
    ).to(device)

    # Prepare latents
    latents = torch.randn(
        1, 4, latent_h, latent_w,
        generator=generator, device=device, dtype=dtype,
    )

    # Setup scheduler
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps
    latents = latents * pipe.scheduler.init_noise_sigma

    # Regional denoising loop
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

        # Unconditional noise prediction
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

        # Per-character conditional predictions blended by soft masks
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

        # Classifier-Free Guidance
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # Scheduler step
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # Decode latents to image
    latents = latents / pipe.vae.config.scaling_factor
    with torch.no_grad():
        image = pipe.vae.decode(latents).sample
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]

    _log.info("[IMAGE GEN] Regional generation complete — %d regions", n_chars)
    return image


# ── Core generation ─────────────────────────────────────────────────────

def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    model_path: str | None = None,
    lora_paths: list[str] | None = None,
    lora_weights: list[float] | None = None,
    mode: str = "normal",
    steps: int = 35,
    guidance_scale: float = 7.5,
    negative_prompt: str | None = None,
    seed: int = -1,
    art_style: str = "custom",
    selected_outfits: dict[str, str] | None = None,
) -> dict:
    """Generate an image. Returns dict with path, prompt analysis, settings used."""
    # Lazy import history functions (still in monolith)
    from .history import record_generation, apply_feedback_learnings, apply_positive_learnings

    # Resolve model
    if model_path is None:
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

    evict_ollama_models()
    pipe = _load_pipeline(model_path)

    # Load LoRAs
    lora_diagnostics = []
    if lora_paths:
        unload_loras(pipe)
        weights = lora_weights or [0.8] * len(lora_paths)
        for lp, lw in zip(lora_paths, weights):
            if Path(lp).exists():
                try:
                    load_lora(pipe, lp, weight=lw)
                except (ValueError, RuntimeError) as e:
                    lora_diagnostics.append({
                        "name": Path(lp).stem,
                        "path": lp,
                        "weight": lw,
                        "trigger_words": [],
                        "loaded": False,
                        "error": str(e),
                    })
                    _log.error("[IMAGE GEN] ✗ LoRA incompatible: %s — %s", Path(lp).stem, e)
                    continue
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

    # Auto-inject missing trigger words
    injected_triggers = []
    if lora_paths:
        tw_data = _load_trigger_words()
        prompt_lower = prompt.lower()
        selected_outfits = selected_outfits or {}

        char_dir = MODELS_DIR / "characters"
        style_dir = MODELS_DIR / "styles"
        char_lora_stems = set()
        style_lora_stems = set()
        for lp in lora_paths:
            try:
                parent = Path(lp).parent.resolve()
                if parent == char_dir.resolve():
                    char_lora_stems.add(Path(lp).stem)
                elif parent == style_dir.resolve():
                    style_lora_stems.add(Path(lp).stem)
            except Exception:
                pass
        multi_char_mode = len(char_lora_stems) >= 2

        for lp in lora_paths:
            if not Path(lp).exists():
                continue
            stem = Path(lp).stem
            if stem in style_lora_stems:
                _log.info("[IMAGE GEN] Skipping auto-inject for style LoRA %s (works via adapter weights)", stem)
                continue
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

    # Strip A1111/ComfyUI-style <lora:...> tags
    full_prompt = re.sub(r'<lora:[^>]+>', '', full_prompt)
    full_prompt = re.sub(r',\s*,', ',', full_prompt)
    full_prompt = re.sub(r'\s{2,}', ' ', full_prompt).strip(', ')

    # Auto-insert BREAK tokens before parenthesized character blocks
    if 'BREAK' not in full_prompt:
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

        paren_blocks = list(re.finditer(r'\([^)]+\)', full_prompt))
        insert_positions: list[int] = []
        for pb in paren_blocks:
            block_text = pb.group(0).lower()
            for code in _all_trigger_codes:
                if code in block_text:
                    insert_positions.append(pb.start())
                    break

        if len(insert_positions) >= 2:
            modified = full_prompt
            for pos in sorted(insert_positions, reverse=True):
                if pos > 0:
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

        # Check for regional multi-character mode
        regional_parsed = _parse_regional_sections(full_prompt)
        use_regional = regional_parsed is not None

        if use_regional:
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
        _update_progress(active=True, type="image", current_step=steps,
                         total_steps=steps, message="Complete!")

        def _clear_progress():
            time.sleep(3)
            _update_progress(active=False, type="idle", current_step=0,
                             total_steps=0, message="")
        threading.Thread(target=_clear_progress, daemon=True).start()


# ── Fast noisy preview ──────────────────────────────────────────────────

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
    """Fast noisy preview at small resolution (256×256) with the same seed."""
    from PIL import Image as PILImage
    from .history import apply_feedback_learnings, apply_positive_learnings

    preview_w = 256
    preview_h = 256

    if seed < 0:
        seed = torch.randint(0, 2**32, (1,)).item()

    model_path = _resolve_model_path(model_path)
    if model_path is None:
        return {"status": "error", "error": "No model files found in models/ directory"}
    if not Path(model_path).exists():
        return {"status": "error", "error": f"Model not found: {model_path}"}

    evict_ollama_models()
    pipe = _load_pipeline(model_path)

    if lora_paths:
        unload_loras(pipe)
        weights = lora_weights or [0.8] * len(lora_paths)
        for lp, lw in zip(lora_paths, weights):
            if Path(lp).exists():
                try:
                    load_lora(pipe, lp, weight=lw)
                except (ValueError, RuntimeError) as e:
                    _log.error("[IMAGE GEN] ✗ Preview LoRA incompatible: %s — %s", Path(lp).stem, e)

    # Build prompt with trigger word injection
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

    full_prompt = re.sub(r'<lora:[^>]+>', '', full_prompt)
    full_prompt = re.sub(r',\s*,', ',', full_prompt)
    full_prompt = re.sub(r'\s{2,}', ' ', full_prompt).strip(', ')

    # Auto-insert BREAK tokens
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

    total_steps = max(preview_steps + 2, 8)
    preview_steps = max(2, min(preview_steps, total_steps - 1))

    generator = torch.Generator(device="cuda").manual_seed(seed)

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

    display_w = max(256, preview_w)
    display_h = max(256, preview_h)
    if preview_img.size != (display_w, display_h):
        preview_img = preview_img.resize((display_w, display_h), PILImage.LANCZOS)

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


# ── Two-pass generation ─────────────────────────────────────────────────

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
    """Two-pass generation: quick noisy preview, then full-res render."""
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

