import asyncio
import httpx
import json
import os
from typing import Optional
from app.rag import get_rag_context

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/chat"
MODEL = "gemma4:e4b"

ORCHESTRATOR_SYSTEM_PROMPT = """You are a medical education orchestrator. Your sole job is to decompose a medical topic or assignment into a precise, ordered list of micro-objectives for a medical student to learn.

Rules:
- Generate between 4 and 8 micro-objectives only
- Each objective must be specific and measurable (start with a verb: Identify, Explain, Differentiate, List, Describe, Calculate, Interpret)
- Each objective must be scoped — do NOT include topics outside the core concept
- Output ONLY valid JSON — no explanation, no markdown, no preamble
- Format exactly as shown below

Output format:
{
  "topic": "<topic name>",
  "objectives": [
    {"id": 1, "verb": "Identify", "objective": "full objective text"},
    {"id": 2, "verb": "Explain", "objective": "full objective text"}
  ]
}"""


async def generate_objectives(topic: str, assignment_text: Optional[str] = None) -> dict:
    user_content = f"Topic: {topic}"
    if assignment_text:
        user_content += f"\n\nAssignment content to decompose:\n{assignment_text}"

    rag_context = await asyncio.to_thread(get_rag_context, topic)
    if rag_context:
        user_content = f"{rag_context}\n\nUsing the context above if relevant, {user_content}"

    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()

    raw = data["message"]["content"].strip()

    # Strip markdown code fences if model adds them
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)