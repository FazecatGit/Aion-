"""
Text-to-speech using Kokoro ONNX — separated from api.py for modularity.
"""

import os
import logging
import tempfile
from typing import Optional

_log = logging.getLogger("tts_kokoro")
_kokoro_instance = None


def _get_kokoro():
    """Lazy-load the Kokoro TTS engine."""
    global _kokoro_instance
    if _kokoro_instance is None:
        from kokoro_onnx import Kokoro
        _kokoro_instance = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    return _kokoro_instance


def generate_speech(
    text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    lang: str = "en-us",
) -> dict:
    """
    Generate speech audio from text.

    Returns dict with 'path' (WAV file) or 'error'.
    """
    try:
        import soundfile as sf
    except ImportError:
        return {"status": "error", "error": "soundfile not installed. Run: pip install soundfile"}

    try:
        kokoro = _get_kokoro()
    except ImportError:
        return {"status": "error", "error": "kokoro-onnx not installed. Run: pip install kokoro-onnx"}

    samples, sample_rate = kokoro.create(
        text,
        voice=voice,
        speed=speed,
        lang=lang,
    )

    if samples is None or len(samples) == 0:
        return {"status": "error", "error": "No audio generated"}

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, samples, sample_rate)
    _log.info("[TTS] Generated %d samples at %d Hz → %s", len(samples), sample_rate, tmp.name)

    return {"status": "ok", "path": tmp.name}
