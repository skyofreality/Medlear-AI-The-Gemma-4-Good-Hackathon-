import asyncio
import io
import os
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