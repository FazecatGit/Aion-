"""Prompt analysis, encoding, validation, vocabulary expansion, and multi-character structuring."""

import math
from pathlib import Path
import re
import logging

import torch
from diffusers import StableDiffusionXLPipeline

from .core import MODELS_DIR, _log, _loaded_pipelines, _active_model
from .lora import (
    _load_trigger_words, parse_outfit_groups,
    parse_trigger_word_entry, build_trigger_prompt, format_trigger_word,
)


# ── Fuzzy matching for character validation ──────────────────────────────

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


# ── Character validation ────────────────────────────────────────────────

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


# ── Token counting & prompt encoding ────────────────────────────────────

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


# ── Prompt analysis ──────────────────────────────────────────────────────

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


# ── Vocabulary Expansion ─────────────────────────────────────────────────

_VOCAB_EXPANSION: dict[str, list[str]] = {
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
    "skilled": ["masterful", "expert", "proficient", "adept", "virtuoso"],
    "devout": ["pious", "reverent", "faithful", "spiritual", "godly"],
    "mischievous": ["playful", "naughty", "impish", "roguish", "cheeky"],
    "passionate": ["ardent", "fervent", "intense", "fiery", "zealous"],
    "methodical": ["precise", "deliberate", "careful", "thorough", "systematic"],
    "soft": ["gentle", "tender", "delicate", "plush", "cushy", "velvety", "silky", "downy", "satin-like"],
    "blush": ["romantic blush", "loving flush", "affectionate glow", "tender blush", "adoring flush", "charmed", "enamored", "smitten", "heartfelt blush", "devoted flush"],
    "nervous": ["coy", "anxious", "timid", "shy", "flustered", "uneasy", "apprehensive", "jittery", "restless", "fidgety", "sheepish", "bashful", "hesitant", "introverted"],
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
    "dark": ["shadowy", "dimly lit", "noir", "tenebrous", "deep shadows"],
    "bright": ["luminous", "radiant", "glowing", "vibrant", "brilliant", "sunlit", "dazzling", "shining", "illuminated", "sparkling"],
    "scary": ["terrifying", "menacing", "ominous", "eldritch", "horrifying"],
    "cool": ["stylish", "composed", "sleek", "suave", "effortlessly chic"],
    "dramatic": ["cinematic", "high contrast", "chiaroscuro", "theatrical", "imposing composition"],
    "warm": ["golden light", "amber tones", "cozy atmosphere", "sunset hues", "inviting glow"],
    "cold": ["blue tones", "icy atmosphere", "frigid", "frosty", "glacial light"],
    "cozy": ["warm and inviting", "snug atmosphere", "intimate haven", "comfortable surroundings", "homey vibe", "Refined comfort", "soft lighting", "relaxed ambiance", "welcoming environment", "pleasantly warm", "cuddle-worthy", "hygge-inspired", "Tranquil"],
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
    "attacking": ["lunging forward", "delivering a strike", "unleashing an assault", "cleaving through", "offensive charge"],
    "blocking": ["parrying a blow", "raising shield", "defensive ward", "deflecting attack", "braced for impact"],
    "dodging": ["evading nimbly", "sidestepping", "rolling away", "ducking under", "acrobatic evasion"],
    "casting": ["channelling energy", "arcane invocation", "hands glowing with power", "spell weaving", "magical incantation"],
    "slashing": ["sword arc", "blade sweep", "cleaving strike", "horizontal cut", "diagonal slash"],
    "shooting": ["taking aim", "firing weapon", "pulling trigger", "archery draw", "projectile release"],
    "punching": ["throwing a fist", "uppercut", "jab strike", "hook punch", "knuckle impact"],
    "kicking": ["roundhouse kick", "high kick", "spinning kick", "front kick", "dropkick"],
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
    "explosion": ["detonation blast wave", "fiery eruption", "shockwave impact", "debris scattering", "cataclysmic burst"],
    "impact": ["collision force", "shattering on contact", "kinetic strike", "concussive blow", "thunderous crash"],
    "pursuit": ["relentless chase", "hot on the trail", "predator stalking prey", "breakneck pursuit", "desperate flee"],
    "transformation": ["metamorphosis", "shape-shifting", "power awakening", "form evolving", "dramatic change"],
    "summoning": ["conjuring forth", "ritual invocation", "materializing from void", "calling upon power", "ethereal manifestation"],
    "magic": ["arcane", "mystical", "ethereal glow", "sorcery", "enchanted"],
    "fire": ["flames", "blazing", "inferno", "ember glow", "pyroclastic"],
    "water": ["aquatic", "submerged", "ripples", "ocean spray", "crystalline water"],
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
    "closeup": ["extreme close-up", "tight framing", "macro shot", "face portrait", "intimate detail"],
    "portrait": ["head and shoulders", "bust shot", "character portrait", "three-quarter view", "profile shot"],
    "wide": ["wide-angle shot", "panoramic view", "full body framing", "establishing shot", "expansive composition"],
    "above": ["bird's eye view", "top-down perspective", "overhead angle", "aerial viewpoint", "looking down at"],
    "below": ["low angle shot", "worm's eye view", "looking up at", "dramatic low perspective", "ground-level"],
    "side": ["profile view", "lateral perspective", "side-on angle", "orthogonal view", "90-degree angle"],
    "behind": ["rear view", "from behind", "over-the-shoulder", "back-facing", "posterior perspective"],
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


# ── Multi-angle / multi-view prompt structuring ──────────────────────────

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


# ── Multi-character prompt structuring ───────────────────────────────────

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
    char_info = []
    for lp in char_lora_paths[:4]:
        stem = Path(lp).stem
        words = tw_data.get(stem, [])
        parsed_groups = parse_outfit_groups(words)
        primary = parsed_groups["primary"] or stem

        trigger_codes = set()
        trigger_codes.add(primary)
        for _outfit_name, tags in parsed_groups.get("outfits", {}).items():
            if tags:
                code = tags[0]["word"] if isinstance(tags[0], dict) else str(tags[0])
                if code:
                    trigger_codes.add(code)
        all_tw = [parse_trigger_word_entry(w)["word"] for w in words
                  if parse_trigger_word_entry(w)["word"] and parse_trigger_word_entry(w)["word"] != ";"]
        for tw_str in all_tw:
            colon_m = re.match(r'^[^:]+:\s*(\S+)', tw_str)
            if colon_m:
                trigger_codes.add(colon_m.group(1).rstrip(','))

        char_info.append((lp, stem, primary, list(trigger_codes), all_tw))

    # ── Step 2: Extract parenthesized character blocks from the prompt ────
    shared = prompt
    char_blocks: dict[str, str] = {}

    for _lp, stem, _primary, trigger_codes, _all_tw in char_info:
        found = False
        for code in sorted(trigger_codes, key=len, reverse=True):
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

    # ── Step 3: Strip any remaining standalone trigger words from shared ───
    for _lp, stem, _primary, trigger_codes, all_tw in char_info:
        for code in trigger_codes:
            shared = re.sub(r'(?<![a-zA-Z0-9])' + re.escape(code) + r'(?![a-zA-Z0-9])',
                            '', shared, flags=re.IGNORECASE)
        for tw_str in all_tw:
            shared = re.sub(r'(?<![a-zA-Z])' + re.escape(tw_str) + r'(?![a-zA-Z])',
                            '', shared, flags=re.IGNORECASE)

    # Strip any leftover BREAK tokens and position phrases from shared
    shared = re.sub(r'\bBREAK\b', '', shared)
    shared = re.sub(r',?\s*on the (?:left|right)\b', '', shared, flags=re.IGNORECASE)
    shared = re.sub(r',?\s*in the (?:center|background)\b', '', shared, flags=re.IGNORECASE)
    shared = re.sub(r'\d+\s*characters?\b', '', shared, flags=re.IGNORECASE)
    shared = re.sub(r'\bgroup\s+shot\b', '', shared, flags=re.IGNORECASE)

    # Clean up orphaned commas, empty parens, extra whitespace
    shared = re.sub(r'\(\s*[:\d.]*\s*\)', '', shared)
    shared = re.sub(r'\(\s*\)', '', shared)
    shared = re.sub(r',\s*,', ',', shared)
    shared = re.sub(r',\s*,', ',', shared)
    shared = re.sub(r'\s{2,}', ' ', shared)
    shared = shared.strip(', \t\n')

    # ── Step 4: Build BREAK-separated character sections ─────────────────
    parts = []
    for i, (lp, stem, primary, _trigger_codes, _all_tw) in enumerate(char_info):
        pos = positions[i] if i < len(positions) else positions[-1]

        if stem in char_blocks:
            parts.append(f"{char_blocks[stem]}, {pos}")
        else:
            words = tw_data.get(stem, [])
            outfit = selected_outfits.get(stem)
            char_triggers = build_trigger_prompt(words, selected_outfit=outfit) if words else stem
            parts.append(f"{char_triggers}, {pos}")

    # ── Step 5: Assemble final prompt ────────────────────────────────────
    num_chars = len(char_info)
    structured = f"{num_chars} characters, group shot, {shared}"
    for part in parts:
        structured += f" BREAK {part}"

    _log.info("[IMAGE GEN] Structured multi-character prompt: %s", structured[:200])
    return structured


# ── Regional conditioning helpers ────────────────────────────────────────

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
    hidden_1 = enc_1.hidden_states[-2]

    # Text encoder 2 (CLIP-G)
    tokens_2 = pipe.tokenizer_2(
        prompt, padding="max_length", max_length=pipe.tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        enc_2 = pipe.text_encoder_2(tokens_2.input_ids, output_hidden_states=True)
    hidden_2 = enc_2.hidden_states[-2]
    pooled = enc_2[0]

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
    n = len(regions)
    xs = torch.linspace(0, 1, latent_w, device=device, dtype=dtype)

    def _centre(pos: str, idx: int) -> float:
        if pos == "left":
            return 0.25
        elif pos == "right":
            return 0.75
        elif pos == "center":
            return 0.5
        elif pos == "background":
            return 0.5
        else:
            return (idx + 0.5) / n

    if n <= 2:
        sigma = 0.35
    elif n == 3:
        sigma = 0.28
    else:
        sigma = 0.22

    raw_masks: list[torch.Tensor] = []
    centres_used = []
    for idx, r in enumerate(regions):
        if r["position"] == "background":
            weight = torch.ones(latent_w, device=device, dtype=dtype) * 0.5
            centres_used.append(0.5)
        else:
            cx = _centre(r["position"], idx)
            centres_used.append(cx)
            weight = torch.exp(-((xs - cx) ** 2) / (2 * sigma ** 2))
        raw_masks.append(weight)

    _log.info("[IMAGE GEN] Spatial masks: %d regions, sigma=%.2f, centres=%s",
              n, sigma, [f"{c:.2f}" for c in centres_used])

    stacked = torch.stack(raw_masks, dim=0)
    temperature = 3.0 if n <= 2 else 4.0
    normed = torch.softmax(stacked * temperature, dim=0)

    masks = []
    for i in range(n):
        mask = normed[i].unsqueeze(0).unsqueeze(0).unsqueeze(0)
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
    if len(sections) < 3:
        return None

    shared = sections[0].strip()
    regions = []
    for sec in sections[1:]:
        sec = sec.strip()
        if not sec:
            continue
        pos = None
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

    # Auto-assign positions when user didn't specify any
    explicit_count = sum(1 for r in regions if r["position"] is not None)
    if explicit_count < len(regions):
        n = len(regions)
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


def _build_anti_bleed_negative(regions: list[dict]) -> str:
    """Generate per-character anti-bleed negative prompt additions.

    Scans each character's description for distinctive features (hair color,
    eye color) and adds them as negatives for OTHER characters' regions.
    This helps SDXL avoid bleeding features between characters.
    """
    color_features = []
    color_pattern = re.compile(
        r'((?:light |dark )?(?:blonde|pink|blue|red|green|purple|white|black|brown|silver|grey|gray|'
        r'orange|yellow|multicolored|streaked)\s+(?:hair|eyes?))',
        re.IGNORECASE,
    )
    for r in regions:
        features = color_pattern.findall(r["prompt"])
        color_features.append(features)

    anti_bleed_parts = []
    for i, features in enumerate(color_features):
        other_features = []
        for j, other in enumerate(color_features):
            if i != j:
                other_features.extend(other)
        if other_features:
            anti_bleed_parts.extend(other_features)

    if anti_bleed_parts:
        seen = set()
        unique = []
        for f in anti_bleed_parts:
            fl = f.lower()
            if fl not in seen:
                seen.add(fl)
                unique.append(f)
        return ", ".join(unique)
    return ""
