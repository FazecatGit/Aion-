"""Generation history, feedback learnings, and custom prompt modifiers."""

import json
import logging
import re
from pathlib import Path
from datetime import datetime

from brain.config import OUTPUT_DIR
from .prompts import _analyze_prompt

_log = logging.getLogger("image_gen")

# ── History state ─────────────────────────────────────────────────────────
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


