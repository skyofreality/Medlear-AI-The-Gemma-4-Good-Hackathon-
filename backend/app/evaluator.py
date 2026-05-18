import asyncio
import httpx
import json
import logging
from app.session import get_session, get_current_objective, advance_session, set_pending_feedback
from app.rag import get_rag_context
from app.config import OLLAMA_CHAT_URL, MODEL

EVALUATION_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_evaluation",
        "description": "Submit the structured comprehension evaluation for the student's response",
        "parameters": {
            "type": "object",
            "properties": {
                "source_alignment": {
                    "type": "object",
                    "description": "How well the student's answer aligns with the source material",
                    "properties": {
                        "score": {"type": "number", "description": "Score from 0.0 to 1.0"},
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Phrases from student answer that match source material",
                        },
                    },
                    "required": ["score", "evidence"],
                },
                "topic_alignment": {
                    "type": "object",
                    "description": "How well the student's answer stays on the current topic",
                    "properties": {
                        "score": {"type": "number", "description": "Score from 0.0 to 1.0"},
                    },
                    "required": ["score"],
                },
                "objective_alignment": {
                    "type": "object",
                    "description": "How completely the student's answer addresses the learning objective",
                    "properties": {
                        "score": {"type": "number", "description": "Score from 0.0 to 1.0"},
                        "missing_elements": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific concepts or elements the student failed to address",
                        },
                    },
                    "required": ["score", "missing_elements"],
                },
                "year_level": {
                    "type": "object",
                    "description": "Whether the depth of the answer is appropriate for a medical student",
                    "properties": {
                        "score": {"type": "number", "description": "Score from 0.0 to 1.0"},
                        "too_easy": {"type": "boolean", "description": "True if answer is too superficial"},
                        "too_advanced": {"type": "boolean", "description": "True if answer uses concepts beyond expected level"},
                    },
                    "required": ["score", "too_easy", "too_advanced"],
                },
                "answer_correctness": {
                    "type": "object",
                    "description": "Factual accuracy and reasoning quality of the student's answer",
                    "properties": {
                        "score": {"type": "number", "description": "Score from 0.0 to 1.0"},
                        "misconceptions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific factual errors or misconceptions in the answer",
                        },
                    },
                    "required": ["score", "misconceptions"],
                },
                "feedback_focus": {
                    "type": "string",
                    "description": "The single most important gap or misconception to address next. Name the exact concept.",
                },
                "advance": {
                    "type": "boolean",
                    "description": "True only when objective_alignment.score >= 0.7 AND answer_correctness.score >= 0.7",
                },
            },
            "required": [
                "source_alignment", "topic_alignment", "objective_alignment",
                "year_level", "answer_correctness", "feedback_focus", "advance"
            ],
        },
    },
}

EVALUATOR_SYSTEM_PROMPT = """You are a silent medical education evaluator. Read the conversation and score the student's understanding across five independent dimensions.

DIMENSIONS (score each 0.0 to 1.0):
- source_alignment: Does the answer reflect the curriculum source material?
- topic_alignment: Does the answer stay on the current topic?
- objective_alignment: Does the answer fully address the specific learning objective?
- year_level: Is the depth appropriate for a medical student?
- answer_correctness: Is the answer factually correct and well-reasoned? A correct concept expressed in synonymous or less common but clinically valid terminology is still correct. Assess whether the reasoning is sound, not whether the student used the exact expected words.

ADVANCE RULES:
- Set advance=true ONLY when objective_alignment.score >= 0.7 AND answer_correctness.score >= 0.7
- Student who only said hello or has not answered yet: all scores 0.0, advance=false
- Student who parroted a definition without reasoning: objective_alignment.score <= 0.4
- Student who answers in own words with correct reasoning: answer_correctness.score >= 0.75
- The user_content includes how many attempts the student has made on this objective. Factor this into your assessment — a student with many attempts who is still not advancing tells a different story than one on their first try.

FEEDBACK FOCUS:
- Set feedback_focus to the single most important missing element or misconception
- Be specific — name the exact concept, not "needs more detail"

Output ONLY via the submit_evaluation tool call. No explanation in content field."""


def _safe_parse(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _empty_eval_result() -> dict:
    return {
        "source_alignment": {"score": 0.0, "evidence": []},
        "topic_alignment": {"score": 0.0},
        "objective_alignment": {"score": 0.0, "missing_elements": []},
        "year_level": {"score": 0.5, "too_easy": False, "too_advanced": False},
        "answer_correctness": {"score": 0.0, "misconceptions": []},
        "feedback_focus": "Evaluator could not assess yet.",
        "advance": False,
        "score": 0.0,
        "satisfied": False,
        "advanced": False,
        "session_complete": False,
    }


async def evaluate_comprehension(session_id: str) -> dict:
    fallback = _empty_eval_result()

    try:
        session = get_session(session_id)
        if not session:
            return fallback

        current = get_current_objective(session_id)
        if not current:
            return {**fallback, "score": 1.0, "satisfied": True,
                    "advance": True, "session_complete": True,
                    "feedback_focus": "All objectives complete."}

        if current.attempt_count < 3:
            return fallback

        logging.info(
            "Evaluator RAG request session_id=%s retrieval_mode=%s doc_id=%s",
            session_id,
            session.retrieval_mode,
            session.doc_id or "",
        )
        eval_rag = await asyncio.to_thread(
            get_rag_context,
            f"{current.verb} {current.objective}",
            5,
            session.doc_id,
            session.retrieval_mode,
        )

        if eval_rag:
            rag_prefix = (
                f"CURRICULUM CONTEXT:\n{eval_rag}\n\n"
                "Use this to understand what concepts the student was taught. "
                "A medically accurate and well-reasoned answer should score well even if it goes beyond the exact wording.\n\n"
            )
        else:
            rag_prefix = "No source material available. Evaluate based on general medical correctness and depth of understanding.\n\n"

        system_content = rag_prefix + EVALUATOR_SYSTEM_PROMPT

        history = session.conversation_history[current.history_start_index:]
        if not history:
            return fallback

        transcript = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in history
        ])

        gaps_note = ""
        if session.coverage_gaps:
            gaps_note = (
                "\nKnown source material gaps (do not penalise student for incomplete answers on these topics): "
                f"{', '.join(session.coverage_gaps)}\n"
            )

        user_content = (
            f"Current objective: {current.verb} {current.objective}\n"
            f"Student attempts on this objective: {current.attempt_count}\n"
            f"{gaps_note}\n"
            f"Conversation transcript:\n{transcript}\n\n"
            "Evaluate whether the student has demonstrated genuine understanding of the objective above."
        )

        payload = {
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "tools": [EVALUATION_TOOL],
            "options": {"temperature": 0.1, "num_ctx": 8192, "think": False},
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OLLAMA_CHAT_URL, json=payload)
            response.raise_for_status()
            data = response.json()

        msg = data["message"]
        args = None

        if msg.get("tool_calls"):
            raw_args = msg["tool_calls"][0]["function"]["arguments"]
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        else:
            logging.warning("Evaluator: tool_calls missing, trying content fallback")
            content = msg.get("content", "").strip()
            if content:
                try:
                    args = _safe_parse(content)
                except (json.JSONDecodeError, ValueError):
                    logging.warning("Evaluator: content parse failed, returning fallback")
                    return fallback
            else:
                logging.warning("Evaluator: empty content and no tool_calls, returning fallback")
                return fallback

        obj_score = float(args.get("objective_alignment", {}).get("score", 0.0))
        ans_score = float(args.get("answer_correctness", {}).get("score", 0.0))
        composite_score = round((obj_score + ans_score) / 2, 3)

        logging.warning(
            "Evaluator scores session_id=%s source_alignment=%s topic_alignment=%s "
            "objective_alignment=%s year_level=%s answer_correctness=%s "
            "feedback_focus=%r advance=%s",
            session_id,
            args.get("source_alignment", {}).get("score"),
            args.get("topic_alignment", {}).get("score"),
            obj_score,
            args.get("year_level", {}).get("score"),
            ans_score,
            args.get("feedback_focus"),
            args.get("advance"),
        )

        missing_elements = args.get("objective_alignment", {}).get("missing_elements", [])
        misconceptions = args.get("answer_correctness", {}).get("misconceptions", [])
        if missing_elements:
            logging.warning("Evaluator missing_elements=%s", missing_elements)
        if misconceptions:
            logging.warning("Evaluator misconceptions=%s", misconceptions)

        result = {
            "source_alignment": args.get("source_alignment", {"score": 0.0, "evidence": []}),
            "topic_alignment": args.get("topic_alignment", {"score": 0.0}),
            "objective_alignment": args.get("objective_alignment", {"score": 0.0, "missing_elements": []}),
            "year_level": args.get("year_level", {"score": 0.5, "too_easy": False, "too_advanced": False}),
            "answer_correctness": args.get("answer_correctness", {"score": 0.0, "misconceptions": []}),
            "feedback_focus": args.get("feedback_focus", ""),
            "advance": bool(args.get("advance", False)),
            "score": composite_score,
            "satisfied": bool(args.get("advance", False)),
        }

        # Write pending_feedback before advancing so tutor gets it on next turn
        set_pending_feedback(session_id, {
            "feedback_focus": result["feedback_focus"],
            "missing_elements": missing_elements,
            "misconceptions": misconceptions,
        })

        if result["advance"]:
            session_complete = advance_session(session_id, composite_score)
            result["advanced"] = True
            result["session_complete"] = session_complete
        else:
            result["advanced"] = False
            result["session_complete"] = False

        return result

    except json.JSONDecodeError as e:
        logging.error("Evaluator JSON parse error: %s", e)
        fallback["feedback_focus"] = f"Evaluator JSON parse error: {e}"
        return fallback
    except Exception as e:
        logging.error("Evaluator failed: %s", e)
        fallback["feedback_focus"] = f"Evaluator error: {e}"
        return fallback
