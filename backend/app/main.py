import base64
import json
import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from typing import Optional
from app.orchestrator import generate_objectives
from app.session import (
    create_session, get_session, get_current_objective,
    get_session_summary
)
from app.teaching import get_teaching_response, stream_sentences
from app.evaluator import evaluate_comprehension
from app.tts import text_to_speech
from app.stt import transcribe_audio

app = FastAPI(title="MedLearn AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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

class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"

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

@app.get("/api/session/{session_id}/summary")
def session_summary(session_id: str):
    summary = get_session_summary(session_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Session not found")
    return summary

@app.post("/api/stt")
async def stt(audio: UploadFile = File(...)):
    data = await audio.read()
    suffix = os.path.splitext(audio.filename or ".webm")[1] or ".webm"
    text = await transcribe_audio(data, suffix=suffix)
    return {"text": text}

@app.post("/api/tts")
async def tts(req: TTSRequest):
    try:
        audio_bytes = await text_to_speech(req.text, req.voice)
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    async def generate():
        async for sentence in stream_sentences(req.session_id, req.message):
            # Send text sentence immediately
            yield f"data: {json.dumps({'type': 'text', 'sentence': sentence})}\n\n"
            # Synthesize and send audio for this sentence
            try:
                wav = await text_to_speech(sentence)
                yield f"data: {json.dumps({'type': 'audio', 'wav': base64.b64encode(wav).decode()})}\n\n"
            except Exception:
                pass  # audio failure is non-fatal

        # Run evaluator after full response is saved to history
        eval_result = await evaluate_comprehension(req.session_id)
        current = get_current_objective(req.session_id)
        yield f"data: {json.dumps({'type': 'eval', 'evaluation': eval_result, 'current_objective': current.dict() if current else None, 'session_complete': eval_result.get('session_complete', False)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/api/chat")
async def chat(req: ChatRequest):
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    ai_response = await get_teaching_response(req.session_id, req.message)
    eval_result = await evaluate_comprehension(req.session_id)
    current = get_current_objective(req.session_id)

    return {
        "ai_response": ai_response,
        "evaluation": eval_result,
        "current_objective": current.dict() if current else None,
        "session_complete": eval_result.get("session_complete", False)
    }