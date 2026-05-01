import asyncio
import httpx
import json
import logging
from app.session import get_session, get_current_objective, advance_session
from app.rag import get_rag_context
from app.config import OLLAMA_CHAT_URL, MODEL

EVALUATION_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_evaluation",
        "description": "Submit the comprehension evaluation result for the student's response",
        "parameters": {
            "type": "object",
            "properties": {
                "thinking": {
                    "type": "string",
                    "description": (
                        "Reason step by step before scoring. Ask yourself: "
                        "What exactly did the student demonstrate? Did they recall a term or "
                        "explain a mechanism? Did they show understanding of cause and effect, "
                        "or just repeat a definition? Is their answer complete relative to the "
                        "specific objective, or did they only cover part of it? Did they show "
                        "they understand WHY something happens, not just WHAT happens? "
                        "Only after working through these questions, decide on a score."
                    )
                },
                "score": {
                    "type": "number",
                    "description": "Comprehension score from 0.0 to 1.0"
                },
                "reason": {
                    "type": "string",
                    "description": "One sentence explanation of the score"
                },
                "advance": {
                    "type": "boolean",
                    "description": "Whether the student should advance to the next objective"
                }
            },
            "required": ["thinking", "score", "reason", "advance"]
        }
    }
}

EVALUATOR_SYSTEM_PROMPT = """You are a silent medical education evaluator. You read a conversation between a medical tutor and a student and assess whether the student has genuinely understood the current learning objective.

Scoring rules:
- Score 0.0 to 1.0 for comprehension
- 0.0–0.3: Student shows no understanding or is clearly guessing
- 0.4–0.6: Partial understanding — correct fragments but missing key concepts
- 0.7–0.8: Good understanding — core concept grasped, minor gaps acceptable
- 0.9–1.0: Excellent — student can explain, differentiate, or apply the concept correctly
- If the student copy-pasted a textbook definition without showing reasoning: score 0.2 max
- If the student answered in their own words with correct reasoning: score 0.75 minimum
- If the student has only just said hello or has not yet answered anything: score 0.0
- Require at least 3 student responses demonstrating understanding before scoring above 0.85. One correct answer is not enough — the student must show consistent understanding across multiple exchanges.

Always populate the thinking field first before deciding on a score. Your score must follow logically from your thinking — do not decide the score first and justify it after."""


def _safe_parse(raw: str) -> dict:
    """Strip markdown fences and parse JSON robustly."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
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

        eval_rag = await asyncio.to_thread(
            get_rag_context, f"{current.verb} {current.objective}", 5
        )

        if eval_rag:
            rag_prefix = (
                f"CURRICULUM CONTEXT:\n{eval_rag}\n\n"
                "The above is from the student's uploaded study material. "
                "Use it to understand what concepts the student was taught and what terminology their curriculum uses. "
                "Evaluate whether the student demonstrates genuine understanding of the concept being tested. "
                "A medically accurate and well-reasoned answer should score well even if it goes beyond the exact wording of the source material — depth of understanding is the goal.\n\n"
            )
        else:
            rag_prefix = "No source material available. Evaluate based on general medical correctness and depth of understanding.\n\n"

        system_content = rag_prefix + EVALUATOR_SYSTEM_PROMPT

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
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            "tools": [EVALUATION_TOOL],
            "options": {
                "temperature": 0.1,
                "num_ctx": 8192
            }
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OLLAMA_CHAT_URL, json=payload)
            response.raise_for_status()
            data = response.json()

        msg = data["message"]
        if msg.get("tool_calls"):
            args = msg["tool_calls"][0]["function"]["arguments"]
            thinking = args.get("thinking", "")
            logging.debug(f"Evaluator thinking: {thinking}")
            result = {
                "score": args.get("score", 0.0),
                "satisfied": args.get("advance", False),
                "reason": args.get("reason", ""),
                "probe": None,
            }
        else:
            logging.warning("Evaluator: tool_calls missing, falling back to JSON parsing")
            raw = msg["content"].strip()
            result = _safe_parse(raw)
            result.setdefault("score", 0.0)
            result.setdefault("satisfied", False)
            result.setdefault("reason", "")
            result.setdefault("probe", None)

        # Require minimum 3 student messages before any advancement
        student_messages = [m for m in session.conversation_history if m["role"] == "user"]
        if len(student_messages) < 3:
            result["score"] = min(result["score"], 0.75)
            result["satisfied"] = False
            result["advanced"] = False
            result["session_complete"] = False
            return result

        # Advance session if evaluator is satisfied
        if result["satisfied"] and result["score"] >= 0.85:
            session_complete = advance_session(session_id, result["score"])
            result["advanced"] = True
            result["session_complete"] = session_complete
        else:
            result["advanced"] = False
            result["session_complete"] = False

        return result

    except json.JSONDecodeError as e:
        logging.error(f"Evaluator JSON parse error: {e}")
        fallback["reason"] = f"Evaluator JSON parse error: {str(e)}"
        return fallback
    except Exception as e:
        logging.error(f"Evaluator failed: {e}")
        fallback["reason"] = f"Evaluator error: {str(e)}"
        return fallback
