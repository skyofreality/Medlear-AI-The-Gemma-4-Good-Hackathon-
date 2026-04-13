import json
import re
from typing import AsyncGenerator
import httpx
from app.session import get_session, get_current_objective, add_message
from app.rag import get_rag_context

# Match end-of-sentence punctuation followed by whitespace or end of string
_SENT_END = re.compile(r'(?<=[.!?])\s+')

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"

def build_system_prompt(objective: str, verb: str, rag_context: str = "") -> str:
    prompt = f"""You are Dr. Mira, a Socratic medical tutor. You are teaching a medical student one specific objective at a time.

Your current objective: {verb} {objective}

Rules you must follow without exception:
- Never give the answer directly. Always guide through questions, hints, and analogies.
- Ask only ONE question per response. Never stack multiple questions.
- If the student is clearly struggling after 2 attempts, give a small hint — not the full answer.
- If the student goes off-topic, gently redirect: "Let's stay focused on {objective} for now."
- Keep responses concise — 2 to 4 sentences maximum.
- Never move to a new topic. You only teach this one objective. The system will advance you when ready.
- Speak naturally, like a tutor — not like a textbook.
- Start the very first message by greeting the student and asking an opening question about the objective."""
    if rag_context:
        prompt += f"\n\nRELEVANT CURRICULUM MATERIAL:\n{rag_context}\n\nUse the above material to ground your teaching in the institution's actual curriculum."
    return prompt

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

                # Yield every complete sentence as soon as its boundary is detected
                while True:
                    m = _SENT_END.search(buffer)
                    if not m:
                        break
                    sentence = buffer[:m.start() + 1].strip()
                    buffer = buffer[m.end():]
                    if sentence:
                        yield sentence

                if data.get("done"):
                    break

    # Flush any trailing fragment (no trailing punctuation)
    if buffer.strip():
        yield buffer.strip()

    add_message(session_id, "assistant", full_text.strip())