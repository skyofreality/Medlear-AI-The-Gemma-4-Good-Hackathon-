import base64
from dataclasses import dataclass
import json
import logging
import os
import fitz  # PyMuPDF
import chromadb
import httpx
from sentence_transformers import SentenceTransformer
from app.config import OLLAMA_CHAT_URL, MODEL

_VISION_SPARSE_THRESHOLD = 100  # chars; below this, use vision instead of raw text
_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 125
_MIN_CHUNK_SIZE = 700

TOPIC_SPAN_SELECTION_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_topic_span_selection",
        "description": "Select candidate source spans that are directly relevant to the user's requested medical or health education topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "accepted_span_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Span IDs that are directly relevant and useful for generating learning objectives.",
                },
                "rejected_spans": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "span_id": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["span_id", "reason"],
                    },
                },
                "topic_interpretation": {"type": "string"},
                "coverage_notes": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": [
                "accepted_span_ids",
                "rejected_spans",
                "topic_interpretation",
                "coverage_notes",
                "confidence",
            ],
        },
    },
}


@dataclass
class RetrievedSpan:
    span_id: str
    doc_id: str | None
    text: str
    filename: str | None
    page_number: int | None
    section_heading: str | None
    chunk_index: int | None
    char_start: int | None
    char_end: int | None
    distance: float | None

# ── Singletons ────────────────────────────────────────────────────────────────

_embedder: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        chroma_path = os.path.join(base_dir, "chroma_db")
        client = chromadb.PersistentClient(path=chroma_path)
        _collection = client.get_or_create_collection(
            name="medlearn_docs",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection

# ── Core functions ─────────────────────────────────────────────────────────────

def _vision_extract_page(page_b64: str) -> str:
    """Call Ollama vision model on a base64-encoded PNG page image."""
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a medical textbook OCR and diagram analysis assistant. Your job is to extract and describe the content of medical textbook pages with complete accuracy.\n\n"
                    "Follow these rules:\n"
                    "- Preserve all medical terminology exactly as written — do not paraphrase or simplify anatomical terms, drug names, units (e.g. mEq/L, mmHg, IU), or abbreviations (e.g. q.d., p.o., IV)\n"
                    "- For text-heavy pages: extract all readable text in reading order, preserving headings, subheadings, and paragraph structure\n"
                    "- For diagram or figure pages: describe the diagram systematically — name the structure, list all labeled components, describe spatial relationships between parts, and note any arrows or flow directions\n"
                    "- For tables: reproduce the table structure as plain text with clear row/column separation\n"
                    "- For pages mixing text and diagrams: extract text first, then describe any figures below it\n"
                    "- Do not add interpretation, clinical commentary, or information not visible on the page"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Extract all text from this medical document page. "
                    "Also describe any diagrams, charts, tables, or figures clearly. "
                    "Output only the extracted content, no commentary."
                ),
                "images": [page_b64],
            },
        ],
        "stream": False,
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(OLLAMA_CHAT_URL, json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def _looks_like_section_heading(line: str) -> bool:
    """Simple heuristic for textbook section headings."""
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return False
    if len(stripped.split()) > 10:
        return False
    if stripped.endswith("."):
        return False

    letters = [ch for ch in stripped if ch.isalpha()]
    if len(letters) < 3:
        return False

    uppercase_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    title_like = stripped == stripped.title()
    return uppercase_ratio >= 0.75 or title_like


def _detect_section_headings(page_text: str) -> list[tuple[int, str]]:
    headings = []
    cursor = 0
    for line in page_text.splitlines(keepends=True):
        if _looks_like_section_heading(line):
            headings.append((cursor, line.strip()))
        cursor += len(line)
    return headings


def _section_for_offset(headings: list[tuple[int, str]], offset: int) -> str:
    current = ""
    for heading_offset, heading in headings:
        if heading_offset > offset:
            break
        current = heading
    return current


def _choose_chunk_end(text: str, start: int) -> int:
    hard_end = min(start + _CHUNK_SIZE, len(text))
    if hard_end == len(text):
        return hard_end

    min_end = min(start + _MIN_CHUNK_SIZE, hard_end)
    window = text[min_end:hard_end]
    for marker in ("\n\n", "\n", ". ", "; ", ", ", " "):
        idx = window.rfind(marker)
        if idx != -1:
            return min_end + idx + len(marker)
    return hard_end


def _split_page_text(page_text: str) -> list[tuple[str, int, int]]:
    chunks = []
    start = 0
    text_len = len(page_text)

    while start < text_len:
        end = _choose_chunk_end(page_text, start)
        raw = page_text[start:end]
        leading_trimmed = len(raw) - len(raw.lstrip())
        trailing_trimmed = len(raw.rstrip())
        char_start = start + leading_trimmed
        char_end = start + trailing_trimmed
        chunk = page_text[char_start:char_end]

        if chunk:
            chunks.append((chunk, char_start, char_end))

        if end >= text_len:
            break
        next_start = max(end - _CHUNK_OVERLAP, start + 1)
        start = next_start

    return chunks


def _metadata_int(metadata: dict, key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metadata_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_retrieved_spans(results: dict) -> list[RetrievedSpan]:
    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]
    ids = results.get("ids") or [[]]

    spans = []
    for index, text in enumerate(documents[0] if documents else []):
        metadata = metadatas[0][index] if metadatas and metadatas[0] and index < len(metadatas[0]) else {}
        distance = distances[0][index] if distances and distances[0] and index < len(distances[0]) else None
        chroma_id = ids[0][index] if ids and ids[0] and index < len(ids[0]) else ""
        span_id = str(metadata.get("span_id") or chroma_id or "")

        spans.append(RetrievedSpan(
            span_id=span_id,
            doc_id=metadata.get("doc_id"),
            text=text or "",
            filename=metadata.get("filename"),
            page_number=_metadata_int(metadata, "page_number"),
            section_heading=metadata.get("section_heading"),
            chunk_index=_metadata_int(metadata, "chunk_index"),
            char_start=_metadata_int(metadata, "char_start"),
            char_end=_metadata_int(metadata, "char_end"),
            distance=_metadata_float(distance),
        ))

    return spans


def preview_span_for_judge(span: RetrievedSpan, max_chars: int = 300) -> dict:
    preview = " ".join(span.text.split())
    if len(preview) > max_chars:
        preview = preview[:max_chars].rstrip() + "..."
    return {
        "span_id": span.span_id,
        "page_number": span.page_number,
        "section_heading": span.section_heading or "",
        "preview": preview,
    }


def _safe_parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _judge_topic_span_relevance(topic: str, candidate_spans: list[RetrievedSpan]) -> dict:
    previews = [preview_span_for_judge(span) for span in candidate_spans]
    preview_payload = json.dumps(previews, ensure_ascii=False)
    logging.info(
        "Topic span judge preview payload topic=%s candidate_count=%s preview_chars=%s",
        topic,
        len(previews),
        len(preview_payload),
    )

    system_prompt = """You are a medical and health education source selection judge.

Your task is to select source spans that are directly relevant to the user's entered topic and useful for generating learning objectives.

Judge generically across anatomy, physiology, pathology, pharmacology, microbiology, community medicine, public health, public awareness, patient education, and any medical or health education topic.

Accept spans that contain curriculum facts, definitions, mechanisms, steps, classifications, examples, clinical links, public health guidance, or patient education guidance directly relevant to the topic.

Reject spans that are only weakly related, too broad, mostly about another topic, or not useful for learning objectives.

Use only the span previews provided. Do not invent span IDs."""

    user_prompt = f"""User-entered topic:
{topic}

Candidate span previews:
{preview_payload}

Select the spans directly relevant to the topic."""

    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "tools": [TOPIC_SPAN_SELECTION_TOOL],
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(OLLAMA_CHAT_URL, json=payload)
        response.raise_for_status()
        data = response.json()

    msg = data["message"]
    if msg.get("tool_calls"):
        args = msg["tool_calls"][0]["function"]["arguments"]
        if isinstance(args, str):
            return _safe_parse_json(args)
        return args
    return _safe_parse_json(msg.get("content", ""))


def retrieve_topic_spans(
    topic: str,
    doc_id: str | None = None,
    retrieval_mode: str = "knowledge_base",
    candidate_count: int = 20,
    max_spans: int = 8,
) -> dict:
    if retrieval_mode == "general_medical":
        logging.info("Topic span retrieval skipped: retrieval_mode=general_medical")
        return {
            "accepted_spans": [],
            "rejected_spans": [],
            "topic_interpretation": "",
            "coverage_notes": "",
            "confidence": 0.0,
        }

    candidate_spans = query_rag(
        topic,
        n_results=candidate_count,
        doc_id=doc_id,
        retrieval_mode=retrieval_mode,
    )
    candidate_span_ids = [span.span_id for span in candidate_spans]
    logging.info(
        "Topic span candidates topic=%s retrieval_mode=%s doc_id=%s candidate_count=%s candidate_span_ids=%s",
        topic,
        retrieval_mode,
        doc_id or "",
        len(candidate_spans),
        candidate_span_ids,
    )

    if not candidate_spans:
        logging.error(
            "RAG returned zero candidates for doc_id=%s retrieval_mode=%s topic=%s",
            doc_id or "",
            retrieval_mode,
            topic,
        )
        if doc_id:
            debug_doc_chunks(doc_id)
        if retrieval_mode == "uploaded_pdf":
            raise ValueError("No topic-relevant content found in the selected PDF for this topic.")
        logging.warning("No candidate spans found in knowledge_base mode; objective generation will use general medical knowledge")
        return {
            "accepted_spans": [],
            "rejected_spans": [],
            "topic_interpretation": "",
            "coverage_notes": "",
            "confidence": 0.0,
        }

    candidate_by_id = {span.span_id: span for span in candidate_spans}
    selection = _judge_topic_span_relevance(topic, candidate_spans)

    accepted_ids = []
    for span_id in selection.get("accepted_span_ids", []):
        if span_id in candidate_by_id and span_id not in accepted_ids:
            accepted_ids.append(span_id)
        elif span_id not in candidate_by_id:
            logging.warning("Topic span judge returned unknown span_id=%s", span_id)
    accepted_ids = accepted_ids[:max_spans]

    rejected_spans = []
    for item in selection.get("rejected_spans", []):
        span_id = item.get("span_id", "")
        if span_id in candidate_by_id:
            rejected_spans.append({
                "span_id": span_id,
                "reason": item.get("reason", ""),
            })

    logging.info(
        "Topic span judge result topic=%s accepted_span_ids=%s confidence=%s coverage_notes=%s",
        topic,
        accepted_ids,
        selection.get("confidence", 0.0),
        selection.get("coverage_notes", ""),
    )
    for rejected in rejected_spans:
        logging.info(
            "Topic span rejected span_id=%s reason=%s",
            rejected["span_id"],
            rejected["reason"],
        )

    if not accepted_ids:
        logging.warning(
            "RAG returned candidates but judge/validation accepted none. topic=%s retrieval_mode=%s doc_id=%s candidate_span_ids=%s rejected_spans=%s",
            topic,
            retrieval_mode,
            doc_id or "",
            candidate_span_ids,
            rejected_spans,
        )
        fallback_spans = candidate_spans[:3]
        fallback_ids = [span.span_id for span in fallback_spans]
        logging.warning(
            "Falling back to top RAG candidates for topic span selection fallback_span_ids=%s",
            fallback_ids,
        )
        accepted_ids = fallback_ids

    return {
        "accepted_spans": [candidate_by_id[span_id] for span_id in accepted_ids],
        "rejected_spans": rejected_spans,
        "topic_interpretation": selection.get("topic_interpretation", ""),
        "coverage_notes": selection.get("coverage_notes", ""),
        "confidence": _metadata_float(selection.get("confidence")),
    }


def ingest_pdf_vision(file_bytes: bytes, filename: str, doc_id: str) -> dict:
    """
    Hybrid extraction: PyMuPDF text for text-rich pages,
    Gemma4 vision for sparse/scanned pages. Chunks and indexes to ChromaDB.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    chunks = []
    ids = []
    metadatas = []

    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        text = page.get_text().strip()
        if len(text) < _VISION_SPARSE_THRESHOLD:
            mat = fitz.Matrix(150 / 72, 150 / 72)
            pix = page.get_pixmap(matrix=mat)
            img_b64 = base64.b64encode(pix.tobytes("png")).decode()
            try:
                text = _vision_extract_page(img_b64).strip()
            except Exception as e:
                logging.warning(f"Vision OCR failed for sparse page, using empty text fallback: {e}")

        headings = _detect_section_headings(text)
        page_chunks = _split_page_text(text)
        logging.info(
            "Chunked PDF page doc_id=%s page_number=%s chunks_indexed=%s",
            doc_id,
            page_number,
            len(page_chunks),
        )

        for chunk_index, (chunk, char_start, char_end) in enumerate(page_chunks):
            span_id = f"{doc_id}::p{page_number}::c{chunk_index}"
            chunks.append(chunk)
            ids.append(span_id)
            metadatas.append({
                "doc_id": doc_id,
                "span_id": span_id,
                "filename": filename,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "char_start": char_start,
                "char_end": char_end,
                "section_heading": _section_for_offset(headings, char_start),
            })
            logging.info(
                "Prepared source span doc_id=%s page_number=%s span_id=%s chunks_indexed=%s",
                doc_id,
                page_number,
                span_id,
                len(page_chunks),
            )

    doc.close()

    if not chunks:
        return {"chunks_indexed": 0, "filename": filename, "doc_id": doc_id}

    embedder = _get_embedder()
    collection = _get_collection()

    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

    BATCH_SIZE = 5000
    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=chunks[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    logging.info(
        "Indexed PDF doc_id=%s filename=%s chunks_indexed=%s",
        doc_id,
        filename,
        len(chunks),
    )
    return {"chunks_indexed": len(chunks), "filename": filename, "doc_id": doc_id}


def query_rag(
    query: str,
    n_results: int = 3,
    doc_id: str | None = None,
    retrieval_mode: str = "knowledge_base",
) -> list[RetrievedSpan]:
    """Return the top-n most similar chunks for a query."""
    if retrieval_mode == "general_medical":
        logging.info("RAG skipped: retrieval_mode=general_medical")
        return []
    if retrieval_mode == "uploaded_pdf" and not doc_id:
        raise ValueError("uploaded_pdf retrieval requires doc_id")
    if retrieval_mode == "knowledge_base":
        logging.info("Using knowledge_base mode for RAG retrieval")
    elif retrieval_mode == "uploaded_pdf":
        logging.info("Using uploaded_pdf mode for RAG retrieval doc_id=%s", doc_id)
    elif retrieval_mode != "uploaded_pdf":
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")

    collection = _get_collection()
    collection_count = collection.count()
    where_filter = {"doc_id": doc_id} if retrieval_mode == "uploaded_pdf" else None
    logging.info(
        "RAG query starting retrieval_mode=%s doc_id=%s collection_count=%s where_filter_applied=%s where_filter=%s query=%s n_results=%s",
        retrieval_mode,
        doc_id or "",
        collection_count,
        bool(where_filter),
        where_filter or {},
        query,
        n_results,
    )
    if collection_count == 0:
        logging.warning(
            "RAG query skipped because collection is empty retrieval_mode=%s doc_id=%s",
            retrieval_mode,
            doc_id or "",
        )
        return []

    embedder = _get_embedder()
    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()

    query_kwargs = {
        "query_embeddings": query_embedding,
        "n_results": min(n_results, collection_count),
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)
    returned_documents = (results.get("documents") or [[]])[0]
    returned_metadatas = (results.get("metadatas") or [[]])[0]
    returned_ids = (results.get("ids") or [[]])[0]
    returned_doc_ids = [
        metadata.get("doc_id") if isinstance(metadata, dict) else None
        for metadata in returned_metadatas
    ]
    returned_span_ids = [
        metadata.get("span_id") if isinstance(metadata, dict) else None
        for metadata in returned_metadatas
    ]
    logging.info(
        "RAG query returned document_count=%s ids=%s metadata_doc_ids=%s metadata_span_ids=%s",
        len(returned_documents),
        returned_ids,
        returned_doc_ids,
        returned_span_ids,
    )
    for index, document in enumerate(returned_documents):
        preview = " ".join((document or "").split())[:150]
        logging.info(
            "RAG returned document preview index=%s id=%s doc_id=%s span_id=%s preview=%r",
            index,
            returned_ids[index] if index < len(returned_ids) else "",
            returned_doc_ids[index] if index < len(returned_doc_ids) else "",
            returned_span_ids[index] if index < len(returned_span_ids) else "",
            preview,
        )

    spans = _build_retrieved_spans(results)
    for span in spans:
        logging.info(
            "Retrieved span span_id=%s page_number=%s distance=%s",
            span.span_id,
            span.page_number,
            span.distance,
        )
    return spans


def format_rag_context(spans: list[RetrievedSpan]) -> str:
    if not spans:
        return ""

    parts = ["---CURRICULUM CONTEXT---"]
    for span in spans:
        distance = f"{span.distance:.4f}" if span.distance is not None else ""
        parts.append(
            f"[span_id={span.span_id}, page={span.page_number or ''}, "
            f"section={span.section_heading or ''}, distance={distance}]"
        )
        parts.append(span.text)
        parts.append("---")
    return "\n".join(parts)


def get_rag_context(
    query: str,
    n_results: int = 3,
    doc_id: str | None = None,
    retrieval_mode: str = "knowledge_base",
) -> str:
    """Return formatted curriculum context for a query, or '' if collection is empty."""
    spans = query_rag(
        query,
        n_results=n_results,
        doc_id=doc_id,
        retrieval_mode=retrieval_mode,
    )
    return format_rag_context(spans)


def get_rag_context_multi(
    queries: list[str],
    n_results_per_query: int = 5,
    doc_id: str | None = None,
    retrieval_mode: str = "knowledge_base",
) -> str:
    """Run multiple queries, deduplicate by exact content match, return combined context string."""
    seen = set()
    deduped_spans = []
    for query in queries:
        for span in query_rag(
            query,
            n_results=n_results_per_query,
            doc_id=doc_id,
            retrieval_mode=retrieval_mode,
        ):
            dedupe_key = span.span_id or span.text
            if dedupe_key not in seen:
                seen.add(dedupe_key)
                deduped_spans.append(span)
    return format_rag_context(deduped_spans)


def debug_doc_chunks(doc_id: str) -> None:
    """Log Chroma metadata details for a specific doc_id without changing retrieval."""
    try:
        collection = _get_collection()
        total_count = collection.count()
        logging.info(
            "RAG debug_doc_chunks start doc_id=%s total_collection_count=%s",
            doc_id,
            total_count,
        )
        if total_count == 0:
            logging.warning("RAG debug_doc_chunks collection is empty doc_id=%s", doc_id)
            return

        matching_all = collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )
        logging.info(
            "RAG debug_doc_chunks matching_count=%s doc_id=%s",
            len(matching_all.get("ids") or []),
            doc_id,
        )

        matching = collection.get(
            where={"doc_id": doc_id},
            include=["metadatas", "documents"],
            limit=min(total_count, 10),
        )
        matching_ids = matching.get("ids") or []
        matching_metadatas = matching.get("metadatas") or []
        matching_documents = matching.get("documents") or []
        logging.info(
            "RAG debug_doc_chunks matching_sample_count=%s matching_ids=%s",
            len(matching_ids),
            matching_ids,
        )
        for index, metadata in enumerate(matching_metadatas[:5]):
            preview = ""
            if index < len(matching_documents):
                preview = " ".join((matching_documents[index] or "").split())[:150]
            logging.info(
                "RAG debug_doc_chunks metadata_sample index=%s metadata=%s preview=%r",
                index,
                metadata,
                preview,
            )

        all_sample = collection.get(
            include=["metadatas"],
            limit=min(total_count, 100),
        )
        available_doc_ids = sorted({
            metadata.get("doc_id")
            for metadata in (all_sample.get("metadatas") or [])
            if isinstance(metadata, dict) and metadata.get("doc_id")
        })
        logging.info(
            "RAG debug_doc_chunks available_doc_ids_sample=%s",
            available_doc_ids,
        )
    except Exception:
        logging.exception("RAG debug_doc_chunks failed doc_id=%s", doc_id)


def is_rag_available(
    doc_id: str | None = None,
    retrieval_mode: str = "knowledge_base",
) -> bool:
    """Returns True if ChromaDB has any documents indexed."""
    try:
        if retrieval_mode == "general_medical":
            logging.info("RAG availability skipped: retrieval_mode=general_medical")
            return False
        if retrieval_mode == "uploaded_pdf" and not doc_id:
            raise ValueError("uploaded_pdf retrieval requires doc_id")
        if retrieval_mode == "knowledge_base":
            logging.info("Using knowledge_base mode for RAG availability check")
        elif retrieval_mode == "uploaded_pdf":
            logging.info("Using uploaded_pdf mode for RAG availability check doc_id=%s", doc_id)
        else:
            raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")

        collection = _get_collection()
        if retrieval_mode == "knowledge_base":
            return collection.count() > 0
        result = collection.get(where={"doc_id": doc_id}, limit=1)
        return bool(result.get("ids"))
    except Exception as e:
        logging.warning(f"RAG availability check failed: {e}")
        return False
