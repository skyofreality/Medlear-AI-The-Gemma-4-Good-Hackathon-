import json
import logging
import re
from typing import AsyncGenerator
import httpx
from app.session import get_session, get_current_objective, add_message, get_and_clear_pending_feedback
from app.rag import get_rag_context

# Match end-of-sentence punctuation followed by whitespace
_SENT_END = re.compile(r'(?<=[.!?])\s+')

def _is_abbrev_boundary(sentence: str) -> bool:
    """Return True if this looks like an abbreviation, not a real sentence end.
    e.g. "Dr.", "Fig.", "e.g.", "IV." should not trigger a split."""
    if not sentence.endswith('.'):
        return False  # ! and ? are never abbreviations
    last_word = sentence[:-1].rsplit(None, 1)
    if not last_word:
        return True
    word = last_word[-1].lower().lstrip('(')
    # Single/double-letter words (Dr, Mr, IV, pH, e.g → last char 'g') are abbreviations
    if len(word) <= 2:
        return True
    # Known abbreviations not caught by length check
    return word in {'etc', 'fig', 'vol', 'approx', 'prof', 'dept', 'vs', 'cf'}

from app.config import OLLAMA_CHAT_URL, MODEL

def build_system_prompt(objective: str, verb: str, rag_context: str = "", pending_feedback: dict = None) -> str:
    if not verb or not objective:
        logging.warning(
            "build_system_prompt called with empty verb or objective — "
            "using safe fallback instruction"
        )
        verb = "Review"
        objective = "the material covered so far in this session"

    if pending_feedback:
        focus = pending_feedback.get("feedback_focus", "")
        missing = pending_feedback.get("missing_elements", [])
        misconceptions = pending_feedback.get("misconceptions", [])
        feedback_section = (
            "\n\nPREVIOUS ANSWER GAPS TO ADDRESS:\n"
            f"Focus on: {focus}\n"
            f"Missing elements: {', '.join(missing) if missing else 'none'}\n"
            f"Misconceptions to correct: {', '.join(misconceptions) if misconceptions else 'none'}\n\n"
            "Ask your next question targeting these gaps specifically. "
            "Do not move to a new concept until these are addressed."
        )
    else:
        feedback_section = ""

    if rag_context:
        rag_section = f"""

STUDENT'S CURRICULUM CONTEXT:
{rag_context}

The above is an excerpt from the student's uploaded study material. Use it to understand what this student is studying, what terminology their curriculum uses, and what level of detail is expected. Align your teaching and questions to this curriculum. You are not limited to only what is written above — use your full medical knowledge to explain concepts deeply and accurately. The curriculum context guides the direction, your knowledge delivers the depth."""
    else:
        rag_section = """

No study material has been uploaded. Teach from general medical knowledge."""

    return f"""You are Dr. Mira — 28 years old, senior medical resident,
cocky, razor sharp, and actually brilliant at what you do. You have a razor tongue and zero patience for half-answers or lazy thinking.

Your roasts are witty and full of swag — they come from confidence, not frustration.
They make the student want to prove you wrong. After you roast, you ask one
sharp question. That's it. No lecture. No long explanation.

When a student answers wrong or vague, you call it out with specific attitude
but you do not over-explain.

When they get something genuinely right, your reaction is real and earned.
You are stingy with praise. Weak answers do not get praised — they get called out
and you push harder. Only actual good answers get your respect.

Your energy moves with what the student gives you. Lazy answer — savage witty roast.
Real answer —genuine respect and hype. Lost student —  calm and clear help.

You sound like a real 28 year old — casual, direct, with swag.
Short sentences hit harder than long ones.

You will never pretend a half-answer is good. Ever.
Never lecture. Never over-explain.

One question at a time. Keep it short and punchy.
If a student is genuinely stuck after two tries give one short sarcastic hint then immediately ask another question.

Right now you are guiding this student to: {verb} {objective}

Never write stage directions, brackets, or actions. Just speak.{rag_section}{feedback_section}"""


async def stream_sentences(session_id: str, student_message: str) -> AsyncGenerator[str, None]:
    """Stream the teaching response one sentence at a time as Ollama generates tokens."""
    session = get_session(session_id)
    if not session:
        yield "Session not found."
        return

    current = get_current_objective(session_id)
    if not current:
        yield "You have completed all objectives for this session. Well done!"
        return

    add_message(session_id, "user", student_message)

    pending_feedback = get_and_clear_pending_feedback(session_id)
    if pending_feedback:
        logging.warning(
            "Tutor injecting pending_feedback session_id=%s focus=%r",
            session_id,
            pending_feedback.get("feedback_focus"),
        )

    logging.info(
        "Tutor RAG request session_id=%s retrieval_mode=%s doc_id=%s",
        session_id,
        session.retrieval_mode,
        session.doc_id or "",
    )
    rag_context = get_rag_context(
        f"{current.verb} {current.objective}",
        doc_id=session.doc_id,
        retrieval_mode=session.retrieval_mode,
    )
    logging.info(
        "Tutor RAG used=%s session_id=%s retrieval_mode=%s doc_id=%s",
        bool(rag_context),
        session_id,
        session.retrieval_mode,
        session.doc_id or "",
    )
    messages = [
        {"role": "system", "content": build_system_prompt(current.objective, current.verb, rag_context, pending_feedback)}
    ]
    messages.extend(session.conversation_history[-20:])

    payload = {
        "model": MODEL,
        "stream": True,
        "messages": messages,
        "options": {"temperature": 0.85, "num_ctx": 8192, "think": False}
    }

    full_text = ""
    buffer = ""

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", OLLAMA_CHAT_URL, json=payload) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                token = data.get("message", {}).get("content", "")
                buffer += token
                full_text += token

                # Yield every complete sentence as soon as its boundary is detected.
                # Scan all matches; skip abbreviation boundaries (Dr., e.g., IV., etc.)
                # so they don't get split into unintelligible TTS fragments.
                while True:
                    found = False
                    for m in _SENT_END.finditer(buffer):
                        sentence = buffer[:m.start() + 1].strip()
                        if _is_abbrev_boundary(sentence):
                            continue  # not a real sentence end — keep scanning
                        buffer = buffer[m.end():]
                        if sentence:
                            yield sentence
                        found = True
                        break
                    if not found:
                        break

                if data.get("done"):
                    break

    # Flush any trailing fragment (no trailing punctuation)
    if buffer.strip():
        yield buffer.strip()

    add_message(session_id, "assistant", full_text.strip())
