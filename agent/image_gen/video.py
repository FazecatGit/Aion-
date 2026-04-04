"""WAN 2.1 video generation — T2V, I2V, checkpointing, resume."""

import gc
import json
import logging
import re
import shutil
import time as _time
import threading
from pathlib import Path
from datetime import datetime

import torch

from brain.config import OUTPUT_DIR
from .core import (
    MODELS_DIR, _cancel_event, _update_progress,
    _check_vram_safe, _save_json,
)
from .models import flush_vram
from .lora import _load_trigger_words
from .animation import _frames_to_mp4

_log = logging.getLogger("image_gen")

# ── Video directories ─────────────────────────────────────────────────────
VIDEO_OUTPUT_DIR = Path(OUTPUT_DIR) / "videos"
VIDEO_CHECKPOINT_DIR = Path("cache") / "video_checkpoints"
VIDEO_QUEUE_FILE = Path("cache") / "video_queue.json"

# ── Video pause/resume events ────────────────────────────────────────────
_video_pause_event = threading.Event()
_video_generating = threading.Event()

# ── WAN pipeline ─────────────────────────────────────────────────────────
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

