import asyncio
import io
import numpy as np
import soundfile as sf
from kokoro import KPipeline

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = KPipeline(lang_code='a')
    return _pipeline

def _synthesize(text: str, voice: str) -> bytes:
    """Synchronous synthesis — runs in a thread pool via asyncio.to_thread."""
    pipeline = get_pipeline()
    chunks = [audio for _, _, audio in pipeline(text, voice=voice, speed=0.93)]
    audio = np.concatenate(chunks)
    buffer = io.BytesIO()
    sf.write(buffer, audio, 24000, format="WAV")
    buffer.seek(0)
    return buffer.read()

async def text_to_speech(text: str, voice: str = "af_bella") -> bytes:
    """Offload CPU-bound Kokoro synthesis to a thread so the event loop stays free."""
    return await asyncio.to_thread(_synthesize, text, voice)
