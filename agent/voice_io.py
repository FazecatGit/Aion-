"""
Voice I/O module: Whisper → Ollama → TTS — hands-free coding help.

Pipeline:
  1. Record audio from microphone (sounddevice)
  2. Transcribe with OpenAI Whisper (local, offline)
  3. Route through Aion (query or agent)
  4. Speak response via pyttsx3 (offline TTS)

Dependencies (install when ready):
  pip install openai-whisper sounddevice pyttsx3 numpy scipy
"""

import os
import io
import logging
import tempfile
import wave
from typing import Optional, Tuple

logger = logging.getLogger("voice_io")

# ── Lazy imports with dependency checks ──────────────────────────────────────

_whisper_model = None
_tts_engine = None

def _get_whisper_model(model_name: str = "base"):
    """Lazy-load Whisper model. Runs fully offline after first download."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        logger.info("[VOICE] Loading Whisper model '%s'...", model_name)
        _whisper_model = whisper.load_model(model_name)
        logger.info("[VOICE] Whisper model loaded")
    return _whisper_model


def _get_tts_engine():
    """Lazy-load TTS engine (pyttsx3 — fully offline)."""
    global _tts_engine
    if _tts_engine is None:
        import pyttsx3
        _tts_engine = pyttsx3.init()
        # Sensible defaults: moderate speed, natural pitch
        _tts_engine.setProperty("rate", 175)
        _tts_engine.setProperty("volume", 0.9)
        logger.info("[VOICE] TTS engine initialized")
    return _tts_engine


# ── Core functions ───────────────────────────────────────────────────────────

def record_audio(duration: float = 5.0, sample_rate: int = 16000) -> bytes:
    """Record audio from default microphone. Returns WAV bytes."""
    import sounddevice as sd
    import numpy as np

    logger.info("[VOICE] Recording %ss of audio at %dHz...", duration, sample_rate)
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    logger.info("[VOICE] Recording complete")

    # Convert to WAV bytes
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return buf.getvalue()


def transcribe_audio(
    audio_bytes: bytes,
    model_name: str = "base",
    language: Optional[str] = "en",
) -> dict:
    """Transcribe audio bytes using Whisper. Returns {text, language, segments}.

    Whisper model sizes (trade-off speed vs accuracy):
      tiny   — fastest, ~1GB VRAM, lower accuracy
      base   — good balance, ~1GB VRAM (recommended)
      small  — better accuracy, ~2GB VRAM
      medium — high accuracy, ~5GB VRAM
      large  — best accuracy, ~10GB VRAM
    """
    model = _get_whisper_model(model_name)

    # Write bytes to temp file (Whisper needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = model.transcribe(
            tmp_path,
            language=language,
            fp16=False,  # CPU-safe
        )
        text = result.get("text", "").strip()
        logger.info("[VOICE] Transcribed: '%s'", text[:100])
        return {
            "text": text,
            "language": result.get("language", language),
            "segments": result.get("segments", []),
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def transcribe_file(file_path: str, model_name: str = "base", language: Optional[str] = "en") -> dict:
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    return transcribe_audio(audio_bytes, model_name, language)


def speak_text(text: str, block: bool = True) -> None:
    engine = _get_tts_engine()
    if len(text) > 2000:
        text = text[:2000] + "... (truncated for speech)"
    engine.say(text)
    if block:
        engine.runAndWait()
    logger.info("[VOICE] Spoke %d chars", len(text))


def speak_to_file(text: str, output_path: str) -> str:
    """Save spoken text to a WAV file. Returns the file path."""
    engine = _get_tts_engine()
    if len(text) > 2000:
        text = text[:2000] + "... (truncated for speech)"
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    logger.info("[VOICE] Saved speech to %s", output_path)
    return output_path


# ── High-level pipeline ─────────────────────────────────────────────────────

def voice_to_text(duration: float = 5.0, model_name: str = "base") -> str:
    audio = record_audio(duration)
    result = transcribe_audio(audio, model_name)
    return result["text"]


def text_to_speech(text: str) -> None:
    speak_text(text)
