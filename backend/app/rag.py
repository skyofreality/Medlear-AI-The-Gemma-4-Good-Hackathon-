import os
import fitz  # PyMuPDF
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import Any

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
