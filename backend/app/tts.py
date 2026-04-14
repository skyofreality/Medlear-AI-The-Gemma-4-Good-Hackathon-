import asyncio
import io
import os
import wave
import soundfile as sf
from kokoro_onnx import Kokoro

_kokoro: Kokoro | None = None

def get_kokoro() -> Kokoro:
    global _kokoro
    if _kokoro is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        onnx_path = os.path.join(base_dir, "kokoro-v1.0.onnx")
        voices_path = os.path.join(base_dir, "voices-v1.0.bin")
        _kokoro = Kokoro(onnx_path, voices_path)
    return _kokoro

def _synthesize(text: str, voice: str) -> bytes:
    """Blocking synthesis — run via asyncio.to_thread."""
    kokoro = get_kokoro()
    samples, sample_rate = kokoro.create(text=text, voice=voice, speed=1.0, lang="en-us")
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()

async def text_to_speech(text: str, voice: str = "af_heart") -> bytes:
    return await asyncio.to_thread(_synthesize, text, voice)


def generate_alignment(text: str, audio_bytes: bytes) -> dict:
    """Generate approximate character-level timing from already-synthesized WAV bytes."""
    with wave.open(io.BytesIO(audio_bytes)) as wf:
        duration = wf.getnframes() / wf.getframerate()
    chars = list(text.replace(" ", ""))
    n = len(chars) if chars else 1
    char_duration = duration / n
    return {
        "chars": chars,
        "char_start_times_seconds": [i * char_duration for i in range(n)],
        "char_durations_seconds": [char_duration] * n,
    }


def _synthesize_with_timing(text: str, voice: str) -> dict:
    """Return base64 WAV + approximate character-level timing."""
    audio_bytes = _synthesize(text, voice)

    # Measure actual audio duration from WAV header
    with wave.open(io.BytesIO(audio_bytes)) as wf:
        duration = wf.getnframes() / wf.getframerate()

    # Distribute duration evenly across non-space characters
    import base64
    chars = list(text.replace(" ", ""))
    n = len(chars) if chars else 1
    char_duration = duration / n
    start_times = [i * char_duration for i in range(n)]
    durations = [char_duration] * n

    return {
        "audio_base64": base64.b64encode(audio_bytes).decode(),
        "alignment": {
            "chars": chars,
            "char_start_times_seconds": start_times,
            "char_durations_seconds": durations,
        },
    }


async def text_to_speech_with_timing(text: str, voice: str = "af_heart") -> dict:
    return await asyncio.to_thread(_synthesize_with_timing, text, voice)