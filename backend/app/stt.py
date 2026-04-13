import asyncio
import os
import tempfile
from faster_whisper import WhisperModel

_model: WhisperModel | None = None

def get_model() -> WhisperModel:
    global _model
    if _model is None:
        # Use local path if the model files are pre-downloaded to backend/whisper-base-en/
        # Otherwise faster-whisper will attempt to download from HuggingFace Hub.
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_path = os.path.join(base_dir, "whisper-base-en")
        model_id = local_path if os.path.isfile(os.path.join(local_path, "model.bin")) else "base.en"
        _model = WhisperModel(model_id, device="cpu", compute_type="int8")
    return _model

def _transcribe(audio_path: str) -> str:
    model = get_model()
    segments, _ = model.transcribe(
        audio_path,
        beam_size=1,        # fastest decoding
        language="en",
        vad_filter=True,    # skip silence chunks
        vad_parameters={"min_silence_duration_ms": 300},
    )
    return " ".join(seg.text.strip() for seg in segments).strip()

async def transcribe_audio(audio_bytes: bytes, suffix: str = ".webm") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        return await asyncio.to_thread(_transcribe, tmp_path)
    finally:
        os.unlink(tmp_path)
