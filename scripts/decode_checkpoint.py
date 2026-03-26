"""Decode final latents from a video checkpoint and export to mp4."""
import torch
import json
import gc
from pathlib import Path
from datetime import datetime

CKPT_DIR = Path("cache/video_checkpoints/20260324_130835_1124125667")
OUTPUT_DIR = Path("generated_images/videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load job metadata
meta = json.load(open(CKPT_DIR / "job.json", encoding="utf-8"))
fps = meta["fps"]
num_frames = meta["num_frames"]
width = meta["width"]
height = meta["height"]
steps = meta["steps"]

print(f"Job: {meta['job_id']}")
print(f"Resolution: {width}x{height}, {num_frames} frames @ {fps} fps, {steps} steps")

# Load the final-step latents (step 20 = fully denoised)
final = torch.load(str(CKPT_DIR / f"latents_step{steps:03d}.pt"), map_location="cpu", weights_only=True)
latents = final["latents"]
print(f"Latents shape: {latents.shape}, dtype: {latents.dtype}")

# Load just the VAE from the WAN model (much smaller than full pipeline)
print("Loading WAN VAE...")
from diffusers import AutoencoderKLWan

# Find local model path
model_path = None
for candidate in [
    Path("models/Wan2.1-T2V-1.3B-Diffusers"),
    Path("models/Wan2.1-T2V-1.3B"),
]:
    if (candidate / "model_index.json").exists():
        model_path = candidate
        break

if model_path is None:
    model_path = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    print(f"Using HuggingFace model: {model_path}")
else:
    print(f"Using local model: {model_path}")

vae = AutoencoderKLWan.from_pretrained(
    model_path,
    subfolder="vae",
    torch_dtype=torch.float32,
)
vae.to("cuda")
vae.enable_tiling()
vae.enable_slicing()
print("VAE loaded on CUDA")

# Decode latents -> video frames
print("Decoding latents through VAE...")
latents = latents.to(dtype=vae.dtype, device="cuda")

# Apply the inverse of the latent normalization the pipeline does
latents_mean = (
    torch.tensor(vae.config.latents_mean)
    .view(1, vae.config.z_dim, 1, 1, 1)
    .to(latents.device, latents.dtype)
)
latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
    latents.device, latents.dtype
)
latents = latents / latents_std + latents_mean

with torch.no_grad():
    video = vae.decode(latents, return_dict=False)[0]

print(f"Decoded video shape: {video.shape}")

# Convert to PIL frames
from diffusers.video_processor import VideoProcessor
processor = VideoProcessor(vae_scale_factor=8)
frames = processor.postprocess_video(video, output_type="pil")[0]
print(f"Got {len(frames)} PIL frames")

# Free VRAM
del vae, latents, video
gc.collect()
torch.cuda.empty_cache()

# Export to mp4
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
mp4_path = str(OUTPUT_DIR / f"{timestamp}_decoded_video.mp4")

try:
    from diffusers.utils import export_to_video
    export_to_video(frames, mp4_path, fps=fps)
    print(f"Video saved: {mp4_path}")
except Exception as e:
    print(f"export_to_video failed: {e}, trying OpenCV...")
    try:
        import cv2
        import numpy as np
        h0, w0 = frames[0].size[1], frames[0].size[0]
        writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w0, h0))
        for frame in frames:
            writer.write(cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Video saved via OpenCV: {mp4_path}")
    except Exception as e2:
        # Last resort: save frames as PNGs
        frame_dir = OUTPUT_DIR / f"{timestamp}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(str(frame_dir / f"frame_{i:04d}.png"))
        print(f"Saved {len(frames)} frames as PNGs in {frame_dir}")

# Save metadata
meta_out = {
    "job_id": meta["job_id"],
    "path": mp4_path,
    "prompt": meta["prompt"],
    "negative_prompt": meta["negative_prompt"],
    "width": width, "height": height,
    "num_frames": len(frames), "fps": fps,
    "steps": steps, "guidance_scale": meta["guidance_scale"],
    "seed": meta["seed"],
    "action_lora": meta.get("action_lora"),
    "created_at": datetime.now().isoformat(),
    "decoded_from_checkpoint": True,
}
meta_path = Path(mp4_path).with_suffix(".json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta_out, f, indent=2)
print(f"Metadata saved: {meta_path}")

# Mark checkpoint as completed
meta["status"] = "completed"
meta["output_path"] = mp4_path
with open(CKPT_DIR / "job.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("Done!")
