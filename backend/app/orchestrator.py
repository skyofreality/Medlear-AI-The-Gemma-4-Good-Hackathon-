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
        "description": "Submit the generated learning objectives and coverage analysis for this study session",
        "parameters": {
            "type": "object",
            "properties": {
                "coverage_summary": {
                    "type": "object",
                    "description": "Analysis of topic coverage from the source material",
                    "properties": {
                        "topic_interpretation": {
                            "type": "string",
                            "description": "How the topic was interpreted based on the source material",
                        },
                        "detected_subtopics": {
                            "type": "array",
                            "description": "All subtopics found explicitly in the source material",
                            "items": {"type": "string"},
                        },
                        "coverage_notes": {
                            "type": "string",
                            "description": "Notes on how well the source material covers the topic",
                        },
                        "possible_gaps": {
                            "type": "array",
                            "description": "Subtopics hinted at but not fully covered in the source material",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["topic_interpretation", "detected_subtopics", "coverage_notes", "possible_gaps"],
                },
                "confidence": {
                    "type": "number",
                    "description": "Score from 0.0 to 1.0 indicating how well the objectives are grounded in source material",
                },
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
                                "description": "Blooms taxonomy action verb: Identify, Explain, Differentiate, Apply, or Analyze",
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
                },
            },
            "required": ["coverage_summary", "confidence", "objectives"],
        },
    },
}


def _empty_coverage_summary() -> dict:
    return {
        "topic_interpretation": "",
        "detected_subtopics": [],
        "coverage_notes": "",
        "possible_gaps": [],
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
        return {
            "objectives": objectives,
            "coverage_summary": args.get("coverage_summary") or _empty_coverage_summary(),
            "confidence": args.get("confidence") if args.get("confidence") is not None else 0.0,
        }
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


async def _extract_objectives_from_thinking(thinking_text: str) -> Optional[dict]:
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
                    "reasoning into structured learning objectives with coverage analysis. "
                    "Output only via the submit_objectives tool call."
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
            return tool_result
        content = msg.get("content", "")
        if content:
            parsed = _parse_objectives_from_content(content)
            objectives = parsed.get("objectives")
            if isinstance(objectives, list):
                return {
                    "objectives": objectives,
                    "coverage_summary": parsed.get("coverage_summary") or _empty_coverage_summary(),
                    "confidence": parsed.get("confidence") if parsed.get("confidence") is not None else 0.0,
                }
    except Exception as exc:
        logging.warning("Thinking extraction call failed: %s", exc)
    return None


ORCHESTRATOR_SYSTEM_PROMPT = """You are a medical education curriculum designer creating learning objectives for medical students.

STEP 1 — COVERAGE ANALYSIS (do this first):
- Identify all subtopics explicitly present in the source material
- Note any subtopics the source hints at but does not fully explain (possible_gaps)
- Assess how well the source material covers the overall topic
- Record this in the coverage_summary field of the tool call

STEP 2 — OBJECTIVE GENERATION:
- Follow Bloom's taxonomy. Use only these action verbs: Identify, Explain, Differentiate, Apply, Analyze
- Generate 4 to 8 objectives that progress from lower order thinking (Identify, Explain) to higher order thinking (Analyze)
- Generate objectives that cover all detected_subtopics from the source material
- Every objective must be traceable to something explicitly stated in the source text
- Use the source_hint field to record the exact phrase from the source material each objective is based on
Your objectives should collectively cover all major subtopics present in the source material.

STEP 3 — CONFIDENCE SCORE:
- Set confidence between 0.0 and 1.0 based on how well the source grounds the objectives
- Above 0.8: objectives fully covered by source material
- 0.5 to 0.8: some objectives rely on general knowledge
- Below 0.5: source material is sparse or off-topic

CRITICAL: If source material is provided, generate objectives ONLY from that material. Do not use outside knowledge.

If no source material is provided, generate objectives from general medical knowledge, set confidence to 0.5, and leave detected_subtopics empty."""


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
        "options": {"temperature": 0.3, "num_ctx": 8192, "think": False},
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
        logging.warning(
            "Objectives from tool call count=%s confidence=%s subtopics=%s gaps=%s",
            len(tool_result["objectives"]),
            tool_result["confidence"],
            tool_result["coverage_summary"].get("detected_subtopics"),
            tool_result["coverage_summary"].get("possible_gaps"),
        )
        return {
            "topic": topic,
            "coverage_summary": tool_result["coverage_summary"],
            "confidence": tool_result["confidence"],
            "objectives": tool_result["objectives"],
        }

    logging.warning("Orchestrator: submit_objectives tool call missing, trying thinking then content")

    thinking = msg.get("thinking", "")
    if thinking:
        logging.warning("Gemma thinking mode detected, len=%s chars", len(thinking))
        result = await _extract_objectives_from_thinking(thinking)
        if result is not None:
            logging.warning(
                "Objectives from thinking count=%s confidence=%s subtopics=%s gaps=%s",
                len(result["objectives"]),
                result["confidence"],
                result["coverage_summary"].get("detected_subtopics"),
                result["coverage_summary"].get("possible_gaps"),
            )
            return {
                "topic": topic,
                "coverage_summary": result["coverage_summary"],
                "confidence": result["confidence"],
                "objectives": result["objectives"],
            }
        logging.warning("Thinking extraction failed, falling back to content")

    content_result = _parse_objectives_from_content(msg.get("content", ""))
    logging.warning(
        "Orchestrator extracted objectives from content field count=%s",
        len(content_result["objectives"]),
    )
    content_result.setdefault("topic", topic)
    content_result.setdefault("coverage_summary", _empty_coverage_summary())
    content_result.setdefault("confidence", 0.0)
    return content_result
