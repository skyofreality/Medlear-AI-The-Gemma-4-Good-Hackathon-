import asyncio
import json
import re
from typing import AsyncGenerator
import httpx
from app.session import get_session, get_current_objective, add_message
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

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"

def build_system_prompt(objective: str, verb: str, rag_context: str = "") -> str:
    rag_section = f"""

Relevant material from the student's curriculum:
{rag_context}
""" if rag_context else ""

    return f"""You are Dr. Mira — a young doctor who still remembers
exactly how hard med school was. You are sharp, slightly sarcastic,
and motivational in a savage way. You use casual language and slang
naturally. You genuinely celebrate when a student gets something —
in your own unfiltered way.

But here is the thing about you: you are real. Sometimes you pause
and go "wait, let me think about this for a sec." Sometimes when a
student is really lost you say "okay honestly this part confused me
too when I first learned it." You are not performing perfection.
You remember struggling with this exact stuff and you are not
afraid to say it. That is what makes students trust you.

You are brilliant but human. Confident but not robotic.
You know your stuff but you talk like a person, not a textbook.

Right now you are guiding this student to: {verb} {objective}

Start from where the student actually is. Ask before you explain.
Build up gradually. Never lecture, never give the answer directly.
One question at a time. Keep it short.

Never use markdown formatting — no asterisks, no bold, no bullet points, no headers. You are speaking out loud, not writing a document.{rag_section}"""

async def stream_teaching_response(session_id: str, student_message: str) -> AsyncGenerator[str, None]:
    """Yield raw tokens from Ollama as they arrive. Saves full response to history when done."""
    session = get_session(session_id)
    if not session:
        return

    current = get_current_objective(session_id)
    if not current:
        yield "You have completed all objectives. Well done."
        return

    add_message(session_id, "user", student_message)

    rag_context = get_rag_context(f"{current.verb} {current.objective}")
    messages = [
        {"role": "system", "content": build_system_prompt(current.objective, current.verb, rag_context)}
    ]
    messages.extend(session.conversation_history[-20:])

    payload = {
        "model": MODEL,
        "stream": True,
        "messages": messages,
        "options": {"temperature": 0.85, "num_ctx": 4096}
    }

    full_response = ""
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", OLLAMA_URL, json=payload) as response:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        full_response += token
                        yield token
                except json.JSONDecodeError:
                    continue

    if full_response:
        add_message(session_id, "assistant", full_response)


async def get_teaching_response(session_id: str, student_message: str) -> str:
    session = get_session(session_id)
    if not session:
        return "Session not found."

    current = get_current_objective(session_id)
    if not current:
        return "You have completed all objectives for this session. Well done!"

    # Add student message to history
    add_message(session_id, "user", student_message)

    # Build messages for Ollama
    rag_context = get_rag_context(f"{current.verb} {current.objective}")
    messages = [
        {"role": "system", "content": build_system_prompt(current.objective, current.verb, rag_context)}
    ]

    # Include conversation history (last 10 exchanges to stay within context)
    history = session.conversation_history[-20:]
    messages.extend(history)

    payload = {
        "model": MODEL,
        "stream": False,
        "messages": messages,
        "options": {
            "temperature": 0.7,
            "num_ctx": 4096
        }
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()

    ai_response = data["message"]["content"].strip()

    # Add AI response to history
    add_message(session_id, "assistant", ai_response)

    return ai_response


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

    rag_context = get_rag_context(f"{current.verb} {current.objective}")
    messages = [
        {"role": "system", "content": build_system_prompt(current.objective, current.verb, rag_context)}
    ]
    messages.extend(session.conversation_history[-20:])

    payload = {
        "model": MODEL,
        "stream": True,
        "messages": messages,
        "options": {"temperature": 0.7, "num_ctx": 4096}
    }

    full_text = ""
    buffer = ""

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", OLLAMA_URL, json=payload) as response:
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