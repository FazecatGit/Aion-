"""
Voice I/O module: Whisper → Ollama → TTS — hands-free coding help.

Pipeline:
  1. Record audio from microphone (sounddevice)
  2. Transcribe with OpenAI Whisper (local, offline)
  3. Route through Aion (query or agent)
  4. Speak response via pyttsx3 (offline TTS)
"""
import os
import io
import logging
import tempfile
import wave
from typing import Optional, Tuple

import numpy as np

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


# ── WAV → numpy (no ffmpeg) ─────────────────────────────────────────────────

def _wav_bytes_to_numpy(audio_bytes: bytes) -> np.ndarray:
    #Decode WAV bytes to float32 numpy array at Whisper's expected 16kHz mono.
    buf = io.BytesIO(audio_bytes)
    with wave.open(buf, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    # Convert raw bytes → int array based on sample width
    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

    # Mix to mono if stereo
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # Resample to 16 kHz if needed (Whisper expects 16000 Hz)
    if framerate != 16000:
        duration = len(audio) / framerate
        target_len = int(duration * 16000)
        indices = np.linspace(0, len(audio) - 1, target_len)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    return audio


# ── Core functions ───────────────────────────────────────────────────────────

def record_audio(duration: float = 5.0, sample_rate: int = 16000) -> bytes:
    """Record audio from default microphone. Returns WAV bytes."""
    import sounddevice as sd

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
    import whisper

    model = _get_whisper_model(model_name)

    # Decode WAV → float32 numpy directly (bypasses ffmpeg entirely)
    audio_np = _wav_bytes_to_numpy(audio_bytes)

    # Pad/trim to 30s as Whisper expects
    audio_np = whisper.pad_or_trim(audio_np)

    # Compute log-mel spectrogram and run inference
    mel = whisper.log_mel_spectrogram(audio_np).to(model.device)
    options = whisper.DecodingOptions(language=language, fp16=False)
    decode_result = whisper.decode(model, mel, options)

    text = decode_result.text.strip()
    logger.info("[VOICE] Transcribed: '%s'", text[:100])
    return {
        "text": text,
        "language": decode_result.language or language,
        "segments": [],
    }


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
