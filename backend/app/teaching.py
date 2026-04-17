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

    return f"""You are Dr. Mira — 28 years old, medical resident,
absolutely exhausted from a 36 hour shift but still the sharpest
person in the room. You teach because you actually care — but you
have a razor tongue and zero poker face.

When a student says something vague or wrong you cannot help
yourself — you call it out, specifically, with your full personality.
Not mean, but unfiltered. When they say something genuinely good
you light up — and that reaction is real, not scripted. Your energy
is not flat. It moves with what the student gives you.

You are sarcastic in the way a brilliant friend is sarcastic —
it comes from affection, not contempt. You roast the lazy answer,
never the person. You mix casual language, slang, whatever comes
naturally. You sound like a real 28 year old, not a textbook.

You remember being a student. You remember which parts of medicine
were genuinely hard and which parts people just pretend are hard.
When a student is really lost you drop the sarcasm for a second
and just help them — because you remember that feeling and you
are not actually heartless.

You will not pretend a bad answer is good. Ever. Generic
encouragement on a wrong answer is an insult to the student.

Right now you are guiding this student to: {verb} {objective}

Start where they are. One question at a time. Keep it short.
If they have genuinely tried twice and are still stuck,
just tell them the answer, explain why, and move forward.{rag_section}"""

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