import asyncio
import base64
import io
import json
import os
import random
import re
import traceback
from dotenv import load_dotenv
load_dotenv()
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
from app.teaching import get_teaching_response, stream_sentences, stream_teaching_response
from app.evaluator import evaluate_comprehension
from app.tts import text_to_speech, get_pipeline
from app.stt import transcribe_audio
from app.rag import ingest_pdf_vision, query_rag

app = FastAPI(title="MedLearn AI API")

# ── Change this to test different voices ─────────────────────────────────────
# American female: af_bella, af_sarah, af_heart, af_sky, af_nicole
# American male:   am_adam, am_michael
AVATAR_VOICE = "af_bella"

@app.on_event("startup")
async def warmup():
    """Pre-load the Kokoro model so the first user request isn't slow."""
    await asyncio.to_thread(get_pipeline)

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

class EvaluateRequest(BaseModel):
    session_id: str

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
        traceback.print_exc()
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

@app.post("/api/tts/with-timing")
async def tts_with_timing(req: TTSRequest):
    try:
        import wave
        audio_bytes = await text_to_speech(req.text, req.voice)
        with wave.open(io.BytesIO(audio_bytes)) as wf:
            duration = wf.getnframes() / wf.getframerate()
        chars = list(req.text.replace(" ", ""))
        n = len(chars) if chars else 1
        char_dur = duration / n
        return {
            "audio_base64": base64.b64encode(audio_bytes).decode(),
            "alignment": {
                "chars": chars,
                "char_start_times_seconds": [i * char_dur for i in range(n)],
                "char_durations_seconds": [char_dur] * n,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def clean_for_speech(text: str) -> str:
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'`+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'_+', '', text)
    # Remove stage directions in square brackets e.g. [Dr. Mira sighs]
    text = re.sub(r'\[([^\]]*)\]', '', text)

    # Remove stage directions in parentheses e.g. (sighs deeply)
    text = re.sub(r'\(([^)]*(?:sighs|laughs|pauses|rubs|leans|looks|smiles|frowns|rolls|shakes|nods|gestures|waves|points|crosses)[^)]*)\)', '', text, flags=re.IGNORECASE)

    # Clean up any double spaces left behind
    text = ' '.join(text.split())
    return text.strip()


async def generate(session_id: str, user_message: str):
    """
    Pipelined SSE generator.
    llm_task  — streams raw tokens, buffers into sentences, pushes to sentence_queue
    tts_task  — pulls sentences, synthesizes audio, pushes SSE events
    Both run concurrently via asyncio.gather so the LLM never waits for TTS.
    """
    sentence_queue: asyncio.Queue = asyncio.Queue()
    events: asyncio.Queue = asyncio.Queue()
    SENTINEL = None

    _ABBR = re.compile(r'\b(Dr|Mr|Mrs|Ms|Prof|vs|etc|e\.g|i\.e|mg|mL|kg|mmHg|bpm|Fig|No|Vol)\.$')

    async def llm_task():
        buffer = ""
        try:
            async for token in stream_teaching_response(session_id, user_message):
                buffer += token

                while True:
                    # Require: punctuation + whitespace + uppercase/quote start
                    # This confirms the word before the period is complete and
                    # a new sentence is actually beginning.
                    match = re.search(r'([^.!?]*[.!?])\s+(?=\S)', buffer)
                    if not match:
                        break

                    sentence = match.group(1).strip()
                    buffer = buffer[match.end():].strip()

                    if len(sentence) < 12:
                        buffer = sentence + " " + buffer
                        break

                    if _ABBR.search(sentence):
                        buffer = sentence + " " + buffer
                        break

                    sentence = clean_for_speech(sentence)
                    await events.put(f"data: {json.dumps({'type': 'text', 'content': sentence})}\n\n")
                    await sentence_queue.put(sentence)
        finally:
            remaining = clean_for_speech(buffer)
            if len(remaining) > 12:
                await events.put(f"data: {json.dumps({'type': 'text', 'content': remaining})}\n\n")
                await sentence_queue.put(remaining)
            await sentence_queue.put(SENTINEL)

    async def tts_task():
        while True:
            sentence = await sentence_queue.get()
            if sentence is SENTINEL:
                await events.put(f"data: {json.dumps({'type': 'done'})}\n\n")
                await events.put(SENTINEL)
                break
            try:
                audio_bytes = await text_to_speech(sentence, AVATAR_VOICE)
                audio_b64 = base64.b64encode(audio_bytes).decode()
                await events.put(f"data: {json.dumps({'type': 'audio', 'content': audio_b64, 'text': sentence})}\n\n")
            except Exception as e:
                print(f"TTS error: {e}")
            sentence_queue.task_done()

    llm = asyncio.create_task(llm_task())
    tts = asyncio.create_task(tts_task())

    while True:
        event = await events.get()
        if event is SENTINEL:
            break
        yield event

    await asyncio.gather(llm, tts)


@app.get("/api/stream")
async def stream_chat(session_id: str, message: str):
    return StreamingResponse(
        generate(session_id, message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest):
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        result = await evaluate_comprehension(req.session_id)
        current = get_current_objective(req.session_id)
        result["current_objective"] = current.dict() if current else None
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
                wav = await text_to_speech(sentence)
                yield f"data: {json.dumps({'type': 'audio', 'wav': base64.b64encode(wav).decode(), 'sentence': sentence})}\n\n"
            except Exception:
                pass

        # Run evaluator after full response is saved to history
        eval_result = await evaluate_comprehension(req.session_id)
        current = get_current_objective(req.session_id)
        yield f"data: {json.dumps({'type': 'eval', 'evaluation': eval_result, 'current_objective': current.dict() if current else None, 'session_complete': eval_result.get('session_complete', False)})}\n\n"

        # If the student just advanced, give Dr. Mira a beat to acknowledge it
        # before the student's next message lands on the new objective cold.
        if eval_result.get('advanced') and current and not eval_result.get('session_complete'):
            _transitions = [
                f"Okay, moving on — {current.verb} {current.objective}. Let's see what you've got.",
                f"Nice. Next up: {current.verb} {current.objective}. Same energy.",
                f"Alright, level up. Now we're on: {current.verb} {current.objective}.",
                f"Good. Let's keep going — {current.verb} {current.objective}.",
                f"See? You had it. Next: {current.verb} {current.objective}.",
            ]
            transition = random.choice(_transitions)
            yield f"data: {json.dumps({'type': 'text', 'sentence': transition})}\n\n"
            try:
                wav = await text_to_speech(transition)
                yield f"data: {json.dumps({'type': 'audio', 'wav': base64.b64encode(wav).decode(), 'sentence': transition})}\n\n"
            except Exception:
                pass

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

@app.get("/api/rag/query")
async def rag_query(q: str):
    chunks = await asyncio.to_thread(query_rag, q)
    return {"chunks": chunks}

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