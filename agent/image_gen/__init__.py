"""
Image generation package — modular split of agent/image_generation.py.

Submodules:
    core        — shared globals, progress tracking, VRAM, cancellation
    models      — pipeline loading, model discovery, VRAM eviction
    lora        — LoRA management, trigger words, categories, search
    styles      — art style presets and application
    prompts     — prompt analysis, encoding, validation, vocabulary
    history     — generation history, feedback, custom learnings
    generation  — core image generation, regional conditioning, preview, upscale
    animation   — animated sequences, storyboards, frame editing
    training    — LoRA training pipeline, dataset critique
    video       — WAN 2.1 video generation, checkpointing, resume

Usage:
    from agent.image_gen import generate_image, list_models, flush_vram
"""

# ── core ──────────────────────────────────────────────────────────────────
from .core import (
    MODELS_DIR, SUPPORTED_EXTENSIONS, LORA_EXTENSIONS,
    get_generation_progress, get_gpu_info, cancel_generation,
    _save_json, _update_progress, _check_vram_safe, _check_cancelled,
)

# ── models ────────────────────────────────────────────────────────────────
from .models import (
    list_models, get_active_model, _load_pipeline,
    evict_ollama_models, flush_vram, unload_model, delete_model,
    _resolve_model_path, _is_lora_file,
)

# ── lora ──────────────────────────────────────────────────────────────────
from .lora import (
    _load_trigger_words, _save_trigger_words,
    parse_trigger_word_entry, format_trigger_word,
    parse_outfit_groups, build_trigger_prompt,
    get_trigger_words, get_trigger_words_parsed,
    set_trigger_words, delete_trigger_words,
    get_all_trigger_words, list_loras_by_category,
    load_lora, unload_loras,
    search_characters,
)

# ── styles ────────────────────────────────────────────────────────────────
from .styles import (
    ART_STYLES, STYLE_NAMES,
    get_art_styles, _apply_art_style,
)

# ── prompts ───────────────────────────────────────────────────────────────
from .prompts import (
    _fuzzy_match_score, validate_prompt_characters,
    count_tokens, _chunk_prompt, _encode_long_prompt, _manual_chunk_encode,
    _split_respecting_parens, _analyze_prompt,
    expand_vocabulary, _VOCAB_EXPANSION,
    _structure_multi_angle_prompt, _structure_multi_character_prompt,
    _encode_single_sdxl, _create_spatial_masks,
    _parse_regional_sections, _build_anti_bleed_negative,
)

# ── history ───────────────────────────────────────────────────────────────
from .history import (
    record_generation, get_generation_history, delete_generation_history_entry,
    apply_feedback_learnings, apply_positive_learnings,
    save_custom_negative, save_custom_positive,
    get_feedback_learnings, clear_feedback_learnings,
    submit_feedback,
)

# ── generation ────────────────────────────────────────────────────────────
from .generation import (
    DEFAULT_NEGATIVE, EXPLICIT_NEGATIVE,
    generate_image, _generate_regional,
    _load_img2img_pipeline,
    generate_preview_only, generate_with_preview,
    upscale_image,
)

# ── animation ─────────────────────────────────────────────────────────────
from .animation import (
    generate_animated, regenerate_frame,
    list_animation_jobs, get_animation_job,
    save_animation_state,
    generate_storyboard,
)

# ── training ──────────────────────────────────────────────────────────────
from .training import (
    get_training_status, critique_training_dataset,
    start_lora_training, cancel_training,
)

# ── video ─────────────────────────────────────────────────────────────────
from .video import (
    generate_video, flush_wan_pipeline,
    pause_video_generation, resume_video_generation,
    resume_interrupted_job,
    get_video_queue_status, list_video_checkpoints,
    list_generated_videos, cleanup_checkpoints,
    list_action_loras,
)
