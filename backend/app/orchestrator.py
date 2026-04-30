import asyncio
import httpx
import json
import logging
import os
from typing import Optional
from app.rag import get_rag_context, is_rag_available

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/chat"
MODEL = "gemma4:e4b"

OBJECTIVES_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_objectives",
        "description": "Submit the generated learning objectives for this study session",
        "parameters": {
            "type": "object",
            "properties": {
                "objectives": {
                    "type": "array",
                    "description": "List of 4 to 8 learning objectives",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "integer",
                                "description": "Sequential index of this objective starting from 1"
                            },
                            "verb": {
                                "type": "string",
                                "description": "Blooms taxonomy action verb: Identify, Explain, Differentiate, Apply, Analyze, or Evaluate"
                            },
                            "objective": {
                                "type": "string",
                                "description": "The specific learning objective for the student"
                            },
                            "source_hint": {
                                "type": "string",
                                "description": "3 to 5 word phrase from the source material this objective is based on. Empty string if no source material provided."
                            }
                        },
                        "required": ["id", "verb", "objective", "source_hint"]
                    }
                }
            },
            "required": ["objectives"]
        }
    }
}

ORCHESTRATOR_SYSTEM_PROMPT = """You are a medical education curriculum designer creating learning objectives for medical students.

Follow Bloom's taxonomy. Use only these action verbs: Identify, Explain, Differentiate, Apply, Analyze, Evaluate.

Generate 4 to 8 objectives that progress from lower order thinking (Identify, Explain) to higher order thinking (Analyze, Evaluate).

CRITICAL: If source material is provided below, generate objectives ONLY from that material. Do not use outside knowledge. Every objective must be traceable to something explicitly stated in the source text. Use the source_hint field to record the exact phrase from the source material that each objective comes from.

If no source material is provided, generate objectives from general medical knowledge about the topic."""


async def generate_objectives(topic: str, assignment_text: Optional[str] = None) -> dict:
    user_content = f"Topic: {topic}"
    if assignment_text:
        user_content += f"\n\nAssignment content to decompose:\n{assignment_text}"

    rag_available = await asyncio.to_thread(is_rag_available)
    if not rag_available:
        logging.warning("Orchestrator: ChromaDB is empty — objectives will be generated from general medical knowledge, not from uploaded PDF")
        user_content = f"No PDF uploaded. Generate objectives from general medical knowledge about: {topic}"
    else:
        rag_context = await asyncio.to_thread(get_rag_context, topic, 12)
        user_content = f"SOURCE MATERIAL FROM STUDENT'S UPLOADED PDF:\n{rag_context}\n\nGenerate objectives ONLY from the source material above.\n\n{user_content}"

    logging.warning(f"DEBUG user_content len: {len(user_content)} chars")

    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        "tools": [OBJECTIVES_TOOL],
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        logging.warning(f"DEBUG raw response keys: {list(data.keys())}")
        logging.warning(f"DEBUG msg: {data['message']!r}")

    msg = data["message"]
    if msg.get("tool_calls"):
        args = msg["tool_calls"][0]["function"]["arguments"]
        return {"topic": topic, "objectives": args["objectives"]}
    else:
        logging.warning("Orchestrator: tool_calls missing, falling back to JSON parsing")
        raw = msg["content"].strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        return json.loads(raw)
