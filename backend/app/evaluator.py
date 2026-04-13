import httpx
import json
from app.session import get_session, get_current_objective, advance_session

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"

EVALUATOR_SYSTEM_PROMPT = """You are a silent medical education evaluator. You read a conversation between a medical tutor and a student and assess whether the student has genuinely understood the current learning objective.

You output ONLY valid JSON — no explanation, no markdown, no preamble.

Scoring rules:
- Score 0.0 to 1.0 for comprehension
- 0.0–0.3: Student shows no understanding or is clearly guessing
- 0.4–0.6: Partial understanding — correct fragments but missing key concepts
- 0.7–0.8: Good understanding — core concept grasped, minor gaps acceptable
- 0.9–1.0: Excellent — student can explain, differentiate, or apply the concept correctly
- If the student copy-pasted a textbook definition without showing reasoning: score 0.2 max
- If the student answered in their own words with correct reasoning: score 0.75 minimum
- If the student has only just said hello or has not yet answered anything: score 0.0

Output format:
{"score": 0.0, "satisfied": false, "reason": "one sentence explanation", "probe": "one follow-up question if not satisfied, or null if satisfied"}"""


def _safe_parse(raw: str) -> dict:
    """Strip markdown fences and parse JSON robustly."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        # parts[1] is the content between first pair of fences
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    return json.loads(raw)


async def evaluate_comprehension(session_id: str) -> dict:
    fallback = {
        "score": 0.0,
        "satisfied": False,
        "reason": "Evaluator could not assess yet.",
        "probe": None,
        "advanced": False,
        "session_complete": False
    }

    try:
        session = get_session(session_id)
        if not session:
            return fallback

        current = get_current_objective(session_id)
        if not current:
            return {**fallback, "score": 1.0, "satisfied": True,
                    "reason": "All objectives complete.", "session_complete": True}

        history = session.conversation_history[-10:]
        if not history:
            return fallback

        transcript = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in history
        ])

        user_content = f"""Current objective: {current.verb} {current.objective}

Conversation transcript:
{transcript}

Evaluate whether the student has demonstrated genuine understanding of the objective above."""

        payload = {
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            "options": {
                "temperature": 0.1,
                "num_ctx": 4096
            }
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()

        raw = data["message"]["content"].strip()
        result = _safe_parse(raw)

        # Ensure required keys exist
        result.setdefault("score", 0.0)
        result.setdefault("satisfied", False)
        result.setdefault("reason", "")
        result.setdefault("probe", None)

        # Advance session if evaluator is satisfied
        if result["satisfied"] and result["score"] >= 0.75:
            session_complete = advance_session(session_id, result["score"])
            result["advanced"] = True
            result["session_complete"] = session_complete
        else:
            result["advanced"] = False
            result["session_complete"] = False

        return result

    except json.JSONDecodeError as e:
        fallback["reason"] = f"Evaluator JSON parse error: {str(e)}"
        return fallback
    except Exception as e:
        fallback["reason"] = f"Evaluator error: {str(e)}"
        return fallback