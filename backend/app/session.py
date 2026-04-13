import uuid
from typing import Optional
from pydantic import BaseModel

class Objective(BaseModel):
    id: int
    verb: str
    objective: str
    completed: bool = False
    comprehension_score: float = 0.0

class Session(BaseModel):
    session_id: str
    topic: str
    objectives: list[Objective]
    current_index: int = 0
    conversation_history: list[dict] = []
    completed: bool = False

# In-memory store — will move to SQLite in Phase 4
sessions: dict[str, Session] = {}

def create_session(topic: str, objectives_data: dict) -> Session:
    session_id = str(uuid.uuid4())
    objectives = [
        Objective(**obj) for obj in objectives_data["objectives"]
    ]
    session = Session(
        session_id=session_id,
        topic=topic,
        objectives=objectives
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
    
    session.objectives[session.current_index].completed = True
    session.objectives[session.current_index].comprehension_score = comprehension_score
    session.current_index += 1
    
    if session.current_index >= len(session.objectives):
        session.completed = True
        return True
    return False

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