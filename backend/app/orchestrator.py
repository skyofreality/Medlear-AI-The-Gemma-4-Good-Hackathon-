import asyncio
import httpx
import json
import logging
from typing import Optional
from app.rag import format_rag_context, retrieve_topic_spans
from app.config import OLLAMA_CHAT_URL, MODEL

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
                                "description": "Sequential index of this objective starting from 1",
                            },
                            "verb": {
                                "type": "string",
                                "description": "Blooms taxonomy action verb: Identify, Explain, Differentiate, Apply, Analyze, or Evaluate",
                            },
                            "objective": {
                                "type": "string",
                                "description": "The specific learning objective for the student",
                            },
                            "source_hint": {
                                "type": "string",
                                "description": "3 to 5 word phrase from the source material this objective is based on. Empty string if no source material provided.",
                            },
                        },
                        "required": ["id", "verb", "objective", "source_hint"],
                    },
                }
            },
            "required": ["objectives"],
        },
    },
}

def _parse_objective_tool_arguments(arguments) -> dict:
    if isinstance(arguments, str):
        return json.loads(arguments)
    if isinstance(arguments, dict):
        return arguments
    raise ValueError(
        f"submit_objectives arguments had unsupported type: {type(arguments).__name__}"
    )


def _extract_objectives_from_tool_calls(message: dict) -> Optional[dict]:
    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        function_call = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        if function_call.get("name") != "submit_objectives":
            continue

        args = _parse_objective_tool_arguments(function_call.get("arguments"))
        objectives = args.get("objectives")
        if not isinstance(objectives, list):
            raise ValueError(
                "submit_objectives tool call did not include an objectives list"
            )
        return {"objectives": objectives}
    return None


def _parse_objectives_from_content(content: str) -> dict:
    raw = (content or "").strip()
    if not raw:
        raise ValueError(
            "Objective generation returned empty content and no usable submit_objectives tool call"
        )
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    parsed = json.loads(raw.strip())
    if not isinstance(parsed, dict) or not isinstance(parsed.get("objectives"), list):
        raise ValueError(
            "Objective generation JSON content did not include an objectives list"
        )
    return parsed


async def _extract_objectives_from_thinking(thinking_text: str) -> Optional[list]:
    logging.warning(
        "Extracting objectives from thinking field, len=%s chars", len(thinking_text)
    )
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a structured data extractor. Convert the following "
                    "reasoning into a JSON array of learning objectives. "
                    "Output only valid JSON, no explanation, no markdown."
                ),
            },
            {"role": "user", "content": thinking_text},
        ],
        "tools": [OBJECTIVES_TOOL],
        "options": {"num_ctx": 8192, "think": False},
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OLLAMA_CHAT_URL, json=payload)
            response.raise_for_status()
            data = response.json()
        msg = data.get("message", {})
        if not isinstance(msg, dict):
            return None
        tool_result = _extract_objectives_from_tool_calls(msg)
        if tool_result is not None:
            return tool_result["objectives"]
        content = msg.get("content", "")
        if content:
            parsed = _parse_objectives_from_content(content)
            return parsed.get("objectives")
    except Exception as exc:
        logging.warning("Thinking extraction call failed: %s", exc)
    return None


ORCHESTRATOR_SYSTEM_PROMPT = """You are a medical education curriculum designer creating learning objectives for medical students.

Follow Bloom's taxonomy. Use only these action verbs: Identify, Explain, Differentiate, Apply, Analyze, Evaluate.

Generate 4 to 8 objectives that progress from lower order thinking (Identify, Explain) to higher order thinking (Analyze, Evaluate).

CRITICAL: If source material is provided below, generate objectives ONLY from that material. Do not use outside knowledge. Every objective must be traceable to something explicitly stated in the source text. Use the source_hint field to record the exact phrase from the source material that each objective comes from.

If no source material is provided, generate objectives from general medical knowledge about the topic."""


async def generate_objectives(
    topic: str,
    assignment_text: Optional[str] = None,
    doc_id: Optional[str] = None,
    retrieval_mode: str = "knowledge_base",
) -> dict:
    logging.info(
        "Generating objectives retrieval_mode=%s doc_id=%s",
        retrieval_mode,
        doc_id or "",
    )

    user_content = f"Topic: {topic}"
    if assignment_text:
        user_content += f"\n\nAssignment content to decompose:\n{assignment_text}"

    if retrieval_mode == "general_medical":
        user_content = (
            "No PDF or knowledge base selected. Generate objectives from "
            f"general medical knowledge about: {topic}"
        )
    else:
        span_selection = await asyncio.to_thread(
            retrieve_topic_spans,
            topic,
            doc_id,
            retrieval_mode,
        )
        accepted_spans = span_selection["accepted_spans"]
        logging.info(
            "Objective topic span selection accepted_count=%s rejected_count=%s confidence=%s retrieval_mode=%s doc_id=%s",
            len(accepted_spans),
            len(span_selection["rejected_spans"]),
            span_selection["confidence"],
            retrieval_mode,
            doc_id or "",
        )
        if accepted_spans:
            rag_context = format_rag_context(accepted_spans)
            user_content = (
                "SOURCE MATERIAL FROM TOPIC-RELEVANT CURRICULUM SPANS:\n"
                f"{rag_context}\n\n"
                "Generate objectives ONLY from the source material above.\n\n"
                f"{user_content}"
            )
        else:
            logging.warning(
                "No accepted topic spans found in knowledge_base mode; objectives will be generated from general medical knowledge"
            )
            user_content = (
                "No topic-relevant knowledge base content found. Generate "
                f"objectives from general medical knowledge about: {topic}"
            )

    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "tools": [OBJECTIVES_TOOL],
        "options": {"num_ctx": 8192, "think": False},
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(OLLAMA_CHAT_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        logging.info("Orchestrator raw response keys=%s", list(data.keys()))
        logging.info(
            "Orchestrator message keys=%s",
            list(message.keys()) if isinstance(message, dict) else [],
        )
        logging.info(
            "Orchestrator message.tool_calls exists=%s count=%s",
            bool(isinstance(message, dict) and message.get("tool_calls")),
            len(message.get("tool_calls") or []) if isinstance(message, dict) else 0,
        )

    msg = data.get("message", {})
    if not isinstance(msg, dict):
        raise ValueError("Objective generation response message was missing or invalid")

    tool_result = _extract_objectives_from_tool_calls(msg)
    if tool_result is not None:
        logging.info(
            "Orchestrator extracted objectives from submit_objectives tool call count=%s",
            len(tool_result["objectives"]),
        )
        return {"topic": topic, "objectives": tool_result["objectives"]}

    logging.warning("Orchestrator: submit_objectives tool call missing, trying thinking then content")

    thinking = msg.get("thinking", "")
    if thinking:
        logging.warning("Gemma thinking mode detected, len=%s chars", len(thinking))
        objectives = await _extract_objectives_from_thinking(thinking)
        if objectives is not None:
            logging.warning(
                "Objectives extracted from thinking successfully count=%s", len(objectives)
            )
            return {"topic": topic, "objectives": objectives}
        logging.warning("Thinking extraction failed, falling back to content")

    content_result = _parse_objectives_from_content(msg.get("content", ""))
    logging.warning(
        "Orchestrator extracted objectives from content field count=%s",
        len(content_result["objectives"]),
    )
    content_result.setdefault("topic", topic)
    return content_result
