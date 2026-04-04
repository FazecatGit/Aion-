"""Animation generation — animated sequences, storyboards, frame editing."""

import json
import logging
import math
import re
import shutil
import time
import threading
from pathlib import Path
from datetime import datetime

import torch

from brain.config import OUTPUT_DIR
from .core import (
    MODELS_DIR, _loaded_pipelines, _active_model,
    _cancel_event, _update_progress, _check_cancelled, _check_vram_safe,
    _save_json,
)
from .models import (
    _load_pipeline, _resolve_model_path, evict_ollama_models,
)
from .lora import (
    _load_trigger_words, parse_outfit_groups, build_trigger_prompt,
    load_lora, unload_loras,
)
from .styles import _apply_art_style, ART_STYLES
from .prompts import (
    _structure_multi_angle_prompt, _structure_multi_character_prompt,
    _encode_long_prompt,
)
from .generation import (
    DEFAULT_NEGATIVE, EXPLICIT_NEGATIVE,
    _load_img2img_pipeline,
)
from .history import apply_feedback_learnings, apply_positive_learnings

_log = logging.getLogger("image_gen")

# ── Animation save directory ─────────────────────────────────────────────
_ANIM_SAVES_DIR = Path(OUTPUT_DIR) / "animation_saves"

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


