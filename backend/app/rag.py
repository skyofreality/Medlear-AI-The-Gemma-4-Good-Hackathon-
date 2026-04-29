import base64
import os
import fitz  # PyMuPDF
import chromadb
import httpx
from chromadb.utils.embedding_functions import EmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import Any

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/chat"
VISION_MODEL = "gemma4:e4b"
_VISION_SPARSE_THRESHOLD = 100  # chars; below this, use vision instead of raw text

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

def ingest_pdf(file_bytes: bytes, filename: str) -> dict:
    """Extract text from PDF, chunk it, embed, and store in ChromaDB."""
    # Extract text page by page
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    # Split into 512-token chunks with 50-token overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.split_text(full_text)

    if not chunks:
        return {"chunks_indexed": 0, "filename": filename}

    embedder = _get_embedder()
    collection = _get_collection()

    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    ids = [f"{filename}::{i}" for i in range(len(chunks))]
    metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]

    # ChromaDB max batch size is 5461 — upsert in safe batches
    BATCH_SIZE = 5000
    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=chunks[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    return {"chunks_indexed": len(chunks), "filename": filename}


def _vision_extract_page(page_b64: str) -> str:
    """Call Ollama vision model on a base64-encoded PNG page image."""
    payload = {
        "model": VISION_MODEL,
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
        resp = client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def ingest_pdf_vision(file_bytes: bytes, filename: str) -> dict:
    """
    Hybrid extraction: PyMuPDF text for text-rich pages,
    Gemma4 vision for sparse/scanned pages. Chunks and indexes to ChromaDB.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_text = []

    for page in doc:
        text = page.get_text().strip()
        if len(text) < _VISION_SPARSE_THRESHOLD:
            # Render at 150 DPI and run through vision model
            mat = fitz.Matrix(150 / 72, 150 / 72)
            pix = page.get_pixmap(matrix=mat)
            img_b64 = base64.b64encode(pix.tobytes("png")).decode()
            try:
                pages_text.append(_vision_extract_page(img_b64))
            except Exception:
                pages_text.append(text)
        else:
            pages_text.append(text)

    doc.close()
    full_text = "\n\n".join(pages_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=50, length_function=len
    )
    chunks = splitter.split_text(full_text)

    if not chunks:
        return {"chunks_indexed": 0, "filename": filename}

    embedder = _get_embedder()
    collection = _get_collection()

    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    ids = [f"{filename}::{i}" for i in range(len(chunks))]
    metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]

    BATCH_SIZE = 5000
    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=chunks[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    return {"chunks_indexed": len(chunks), "filename": filename}


def query_rag(query: str, n_results: int = 3) -> list[str]:
    """Return the top-n most similar chunks for a query."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    embedder = _get_embedder()
    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count()),
        include=["documents"],
    )
    return results["documents"][0] if results["documents"] else []


def get_rag_context(query: str) -> str:
    """Return formatted curriculum context for a query, or '' if collection is empty."""
    chunks = query_rag(query)
    if not chunks:
        return ""
    parts = ["---CURRICULUM CONTEXT---"]
    for chunk in chunks:
        parts.append(chunk)
        parts.append("---")
    return "\n".join(parts)
