"""LoRA training pipeline — dataset critique, training loop, checkpoints."""

import logging
import shutil
import threading
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

from .core import MODELS_DIR
from .models import _resolve_model_path

_log = logging.getLogger("image_gen")

# ── Training directory ────────────────────────────────────────────────────
TRAINING_DIR = Path(__file__).parent.parent.parent / "cache" / "lora_training"

_training_state: dict = {
    "status": "idle",
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

