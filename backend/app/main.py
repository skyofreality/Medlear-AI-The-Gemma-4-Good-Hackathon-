import asyncio
import base64
import json
import logging
import os
import traceback
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from app.orchestrator import generate_objectives
from app.session import (
    create_session, get_session, get_current_objective,
    generate_session_summary
)
from app.teaching import stream_sentences
from app.evaluator import evaluate_comprehension
from app.tts import text_to_speech, get_pipeline
from app.stt import transcribe_audio
from app.rag import ingest_pdf_vision
from app.config import CORS_ORIGIN

app = FastAPI(title="MedLearn AI API")

@app.on_event("startup")
async def warmup():
    """Pre-load the Kokoro model so the first user request isn't slow."""
    await asyncio.to_thread(get_pipeline)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TopicRequest(BaseModel):
    topic: str
    assignment_text: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.get("/")
def root():
    return {"status": "MedLearn AI backend running"}

@app.post("/api/session/start")
async def start_session(req: TopicRequest):
    try:
        objectives_data = await generate_objectives(req.topic, req.assignment_text)
        session = create_session(req.topic, objectives_data)
        current = get_current_objective(session.session_id)
        return {
            "session_id": session.session_id,
            "topic": session.topic,
            "total_objectives": len(session.objectives),
            "objectives": [o.dict() for o in session.objectives],
            "current_objective": current.dict() if current else None
        }
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}")
def get_session_state(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    current = get_current_objective(session_id)
    return {
        "session_id": session_id,
        "topic": session.topic,
        "current_index": session.current_index,
        "total_objectives": len(session.objectives),
        "current_objective": current.dict() if current else None,
        "completed": session.completed,
        "objectives": [o.dict() for o in session.objectives]
    }

@app.post("/api/stt")
async def stt(audio: UploadFile = File(...)):
    data = await audio.read()
    suffix = os.path.splitext(audio.filename or ".webm")[1] or ".webm"
    text = await transcribe_audio(data, suffix=suffix)
    return {"text": text}

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    async def generate():
        async for sentence in stream_sentences(req.session_id, req.message):
            # Send text immediately
            yield f"data: {json.dumps({'type': 'text', 'sentence': sentence})}\n\n"
            # Synthesize audio for this sentence, then send it
            try:
                wav, alignment = await text_to_speech(sentence)
                yield f"data: {json.dumps({'type': 'audio', 'wav': base64.b64encode(wav).decode(), 'sentence': sentence, 'alignment': alignment})}\n\n"
            except Exception as e:
                logging.error(f"TTS synth failed for sentence: {e}")

        eval_result = await evaluate_comprehension(req.session_id)
        current = get_current_objective(req.session_id)

        session_summary = ""
        if eval_result.get('session_complete'):
            try:
                session_summary = await generate_session_summary(req.session_id)
            except Exception as e:
                logging.error(f"Session summary generation failed: {e}")

        yield f"data: {json.dumps({'type': 'eval', 'evaluation': eval_result, 'current_objective': current.dict() if current else None, 'session_complete': eval_result.get('session_complete', False), 'session_summary': session_summary})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/api/rag/ingest")
async def rag_ingest(file: UploadFile = File(...)):
    data = await file.read()
    result = await asyncio.to_thread(ingest_pdf_vision, data, file.filename or "upload.pdf")
    return {"message": f"Indexed {result['chunks_indexed']} chunks from {result['filename']}", "chunks_indexed": result['chunks_indexed'], "filename": result['filename']}

