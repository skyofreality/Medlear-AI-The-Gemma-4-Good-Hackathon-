import httpx
import logging
import uuid
from typing import Literal, Optional
from pydantic import BaseModel
from app.config import OLLAMA_CHAT_URL, MODEL

RetrievalMode = Literal["uploaded_pdf", "knowledge_base", "general_medical"]

class Objective(BaseModel):
    id: int = 0
    verb: str
    objective: str
    completed: bool = False
    comprehension_score: float = 0.0
    source_hint: str = ""

class Session(BaseModel):
    session_id: str
    topic: str
    objectives: list[Objective]
    doc_id: Optional[str] = None
    retrieval_mode: RetrievalMode = "knowledge_base"
    current_index: int = 0
    conversation_history: list[dict] = []
    completed: bool = False

# In-memory store — will move to SQLite in Phase 4
sessions: dict[str, Session] = {}

def validate_retrieval_config(
    retrieval_mode: str,
    doc_id: Optional[str] = None,
) -> None:
    if retrieval_mode not in ("uploaded_pdf", "knowledge_base", "general_medical"):
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")
    if retrieval_mode == "uploaded_pdf" and not doc_id:
        raise ValueError("uploaded_pdf retrieval mode requires doc_id")


def create_session(
    topic: str,
    objectives_data: dict,
    doc_id: Optional[str] = None,
    retrieval_mode: RetrievalMode = "knowledge_base",
) -> Session:
    validate_retrieval_config(retrieval_mode, doc_id)
    logging.info(
        "Creating session retrieval_mode=%s doc_id=%s",
        retrieval_mode,
        doc_id or "",
    )
    session_id = str(uuid.uuid4())
    objectives = [
        Objective(**{**obj, "id": i + 1})
        for i, obj in enumerate(objectives_data["objectives"])
    ]
    session = Session(
        session_id=session_id,
        topic=topic,
        objectives=objectives,
        doc_id=doc_id,
        retrieval_mode=retrieval_mode,
    )
    sessions[session_id] = session
    return session

def get_session(session_id: str) -> Optional[Session]:
    return sessions.get(session_id)

def get_current_objective(session_id: str) -> Optional[Objective]:
    session = get_session(session_id)
    if not session or session.completed:
        return None
    return session.objectives[session.current_index]

def add_message(session_id: str, role: str, content: str):
    session = get_session(session_id)
    if session:
        session.conversation_history.append({
            "role": role,
            "content": content
        })

def advance_session(session_id: str, comprehension_score: float) -> bool:
    """Mark current objective complete and move to next. Returns True if session is now fully complete."""
    session = get_session(session_id)
    if not session:
        return False

    if session.completed:
        logging.warning(
            f"advance_session called on completed session "
            f"(session={session_id}) — ignoring"
        )
        return True

    current_obj = session.objectives[session.current_index]
    if current_obj.completed:
        logging.warning(
            f"advance_session called on already-completed objective "
            f"(session={session_id}, index={session.current_index}) — ignoring"
        )
        return session.completed

    session.objectives[session.current_index].completed = True
    session.objectives[session.current_index].comprehension_score = comprehension_score
    session.current_index += 1

    if session.current_index >= len(session.objectives):
        session.completed = True
        return True
    return False

async def generate_session_summary(session_id: str) -> str:
    """
    Uses Gemma 4 to generate a personalized summary of the student's
    session performance. Called when a session completes.
    Returns a summary string.
    """
    session = get_session(session_id)
    if not session:
        return f"You completed your session. Review any objectives marked as needing more work before your next session."

    objectives_text = "\n".join([
        f"- {o.verb} {o.objective}: {'mastered' if o.comprehension_score >= 0.75 else 'needs more work'} (score: {o.comprehension_score:.2f})"
        for o in session.objectives
    ])

    system_prompt = (
        "You are a supportive medical education tutor reviewing a student's study session. "
        "Write a brief, honest, personalized summary of their performance.\n\n"
        "Rules:\n"
        "- 3 to 5 sentences maximum\n"
        "- Be specific — name the actual objectives they struggled with or excelled at, do not speak in generalities\n"
        "- Be encouraging but accurate — do not praise poor performance\n"
        "- Use plain student-friendly language, no jargon about scoring\n"
        "- Do not mention numbers or scores directly\n"
        "- End with one concrete suggestion for what to review next"
    )

    user_message = (
        f"Topic: {session.topic}\n\n"
        f"Objectives and performance:\n{objectives_text}\n\n"
        "Write a personalized summary for this student."
    )

    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "options": {"temperature": 0.4, "num_ctx": 8192},
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OLLAMA_CHAT_URL, json=payload)
            response.raise_for_status()
            data = response.json()
        summary = data["message"]["content"].strip()
        logging.info(f"Session summary generated for session {session_id}")
        return summary
    except Exception as e:
        logging.error(f"Session summary generation failed: {e}")
        return f"You completed your session on {session.topic}. Review any objectives marked as needing more work before your next session."


def get_session_summary(session_id: str) -> dict:
    session = get_session(session_id)
    if not session:
        return {}
    
    mastered = [o for o in session.objectives if o.completed and o.comprehension_score >= 0.75]
    weak = [o for o in session.objectives if o.completed and o.comprehension_score < 0.75]
    
    return {
        "session_id": session_id,
        "topic": session.topic,
        "total_objectives": len(session.objectives),
        "completed_objectives": len([o for o in session.objectives if o.completed]),
        "mastered": [o.objective for o in mastered],
        "needs_review": [o.objective for o in weak],
        "overall_score": round(
            sum(o.comprehension_score for o in session.objectives if o.completed) /
            max(len([o for o in session.objectives if o.completed]), 1), 2
        )
    }
