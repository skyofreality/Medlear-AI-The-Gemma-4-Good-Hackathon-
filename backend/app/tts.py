import asyncio
import io
import re
import numpy as np
import soundfile as sf
from kokoro import KPipeline

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = KPipeline(lang_code='a')
    return _pipeline

def clean_text_for_tts(text: str) -> str:
    # Remove emojis and non-latin unicode symbols
    text = re.sub(r'[^\x00-\x7FÀ-ɏḀ-ỿ]', '', text)
    # Remove markdown characters
    text = re.sub(r'[*_#`~]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

SAMPLE_RATE = 24000

def _synthesize(text: str, voice: str) -> tuple[bytes, dict]:
    """Synchronous synthesis — runs in a thread pool via asyncio.to_thread.

    Returns (wav_bytes, alignment) where alignment is:
      {"words": [...], "wtimes": [ms_start, ...], "wdurations": [ms_dur, ...]}
    Word timings come from Kokoro's per-token start_ts/end_ts (filled in by
    KPipeline.join_timestamps from pred_dur). When tokens are missing
    timestamps, alignment for that chunk is skipped — frontend falls back
    to client-side synthesis.
    """
    pipeline = get_pipeline()
    text = clean_text_for_tts(text)

    audio_chunks: list = []
    words: list[str] = []
    wtimes: list[float] = []
    wdurations: list[float] = []
    offset_s = 0.0

    for result in pipeline(text, voice=voice, speed=0.93):
        audio = result.audio
        if audio is None:
            continue
        chunk_dur_s = len(audio) / SAMPLE_RATE

        tokens = getattr(result, "tokens", None)
        if tokens:
            for tok in tokens:
                start = getattr(tok, "start_ts", None)
                end = getattr(tok, "end_ts", None)
                if start is None or end is None:
                    continue
                w = (getattr(tok, "text", None) or "").strip()
                if not w:
                    continue
                start_ms = (offset_s + float(start)) * 1000.0
                dur_ms = max(50.0, (float(end) - float(start)) * 1000.0)
                words.append(w)
                wtimes.append(start_ms)
                wdurations.append(dur_ms)

        audio_chunks.append(audio)
        offset_s += chunk_dur_s

    if audio_chunks:
        audio_concat = np.concatenate(audio_chunks)
    else:
        audio_concat = np.zeros(0, dtype=np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio_concat, SAMPLE_RATE, format="WAV")
    buffer.seek(0)
    alignment = {"words": words, "wtimes": wtimes, "wdurations": wdurations}
    return buffer.read(), alignment

async def text_to_speech(text: str, voice: str = "af_bella") -> tuple[bytes, dict]:
    """Offload CPU-bound Kokoro synthesis to a thread so the event loop stays free.
    Returns (wav_bytes, alignment_dict).
    """
    return await asyncio.to_thread(_synthesize, text, voice)
