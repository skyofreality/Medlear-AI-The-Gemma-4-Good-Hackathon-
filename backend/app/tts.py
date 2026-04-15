"""
TTS provider module.

Switch engines by changing TTS_PROVIDER below.
Switch Piper voices by changing PIPER_MODEL below.
All other code (main.py, teaching.py, etc.) imports only:
    text_to_speech(text, voice) -> bytes (WAV)
    text_to_speech_with_timing(text, voice) -> dict
    generate_alignment(text, audio_bytes) -> dict
These are stable — provider/voice swaps are fully internal.

Available providers
-------------------
"piper"   — Piper TTS  (~300 ms/sentence, offline, recommended)
"kokoro"  — Kokoro ONNX (~1–3 s/sentence, offline, original)

Piper female voices (place .onnx + .onnx.json in backend/)
-----------------------------------------------------------
"en_US-lessac-medium"      American — clear, neutral (current)
"en_US-amy-medium"         American — warm, natural 9.1
"en_US-hfc_female-medium"  American — crisp, professional - 8.9
"en_GB-jenny_dioco-medium" British  — polished, authoritative - 9
"en_US-ljspeech-high"      American — highest quality, ~10% slower
"en_GB-alba-medium"        Scottish — warm, distinctive

Download any voice (run from backend/ dir):
  curl -L -o <MODEL>.onnx      "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/<REGION>/<NAME>/<QUALITY>/<MODEL>.onnx"
  curl -L -o <MODEL>.onnx.json "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/<REGION>/<NAME>/<QUALITY>/<MODEL>.onnx.json"
"""

import asyncio
import io
import os
import wave

# ─────────────────────────────────────────────────────────────
#  PROVIDER SELECTION  ←  change this one line to switch engine
# ─────────────────────────────────────────────────────────────
TTS_PROVIDER = "piper"   # "piper" | "kokoro"

#  PIPER VOICE  ←  change this one line to switch voice
#  (model file must exist in backend/ as <PIPER_MODEL>.onnx)
PIPER_MODEL = "en_US-amy-medium"
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ══════════════════════════════════════════════════════════════
#  PIPER PROVIDER
# ══════════════════════════════════════════════════════════════
_piper_voice = None

def _get_piper():
    global _piper_voice
    if _piper_voice is None:
        from piper import PiperVoice
        model_path = os.path.join(BASE_DIR, f"{PIPER_MODEL}.onnx")
        _piper_voice = PiperVoice.load(model_path)
    return _piper_voice

def _piper_synthesize(text: str, _voice: str) -> bytes:
    """_voice is unused — Piper voice is determined by the model file."""
    piper = _get_piper()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        piper.synthesize_wav(text, wf)  # sets channels/rate/width automatically
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════
#  KOKORO PROVIDER
# ══════════════════════════════════════════════════════════════
_kokoro_instance = None

def _get_kokoro():
    global _kokoro_instance
    if _kokoro_instance is None:
        from kokoro_onnx import Kokoro
        onnx_path   = os.path.join(BASE_DIR, "kokoro-v1.0.onnx")
        voices_path = os.path.join(BASE_DIR, "voices-v1.0.bin")
        _kokoro_instance = Kokoro(onnx_path, voices_path)
    return _kokoro_instance

def _kokoro_synthesize(text: str, voice: str) -> bytes:
    import soundfile as sf
    kokoro = _get_kokoro()
    samples, sample_rate = kokoro.create(text=text, voice=voice, speed=1.0, lang="en-us")
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════
#  ROUTER  — dispatches to active provider
# ══════════════════════════════════════════════════════════════
_PROVIDERS = {
    "piper":  _piper_synthesize,
    "kokoro": _kokoro_synthesize,
}

def _synthesize(text: str, voice: str) -> bytes:
    fn = _PROVIDERS.get(TTS_PROVIDER)
    if fn is None:
        raise ValueError(f"Unknown TTS_PROVIDER: '{TTS_PROVIDER}'. Choose from: {list(_PROVIDERS)}")
    return fn(text, voice)


# ══════════════════════════════════════════════════════════════
#  PUBLIC API  — stable regardless of provider
# ══════════════════════════════════════════════════════════════

async def text_to_speech(text: str, voice: str = "af_heart") -> bytes:
    """Synthesize text → WAV bytes (async, non-blocking)."""
    return await asyncio.to_thread(_synthesize, text, voice)


def generate_alignment(text: str, audio_bytes: bytes) -> dict:
    """
    Derive approximate character-level timing from WAV duration.
    Works for any provider — only needs the final WAV bytes.
    """
    with wave.open(io.BytesIO(audio_bytes)) as wf:
        duration = wf.getnframes() / wf.getframerate()
    chars = list(text.replace(" ", ""))
    n = len(chars) if chars else 1
    char_dur = duration / n
    return {
        "chars": chars,
        "char_start_times_seconds": [i * char_dur for i in range(n)],
        "char_durations_seconds":   [char_dur] * n,
    }


def _synthesize_with_timing(text: str, voice: str) -> dict:
    import base64
    audio_bytes = _synthesize(text, voice)
    alignment   = generate_alignment(text, audio_bytes)
    return {
        "audio_base64": base64.b64encode(audio_bytes).decode(),
        "alignment":    alignment,
    }

async def text_to_speech_with_timing(text: str, voice: str = "af_heart") -> dict:
    """Synthesize text → {audio_base64, alignment} (async, non-blocking)."""
    return await asyncio.to_thread(_synthesize_with_timing, text, voice)
