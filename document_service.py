"""
document_service.py - Orchestrates the full ingestion + search pipeline.

This is the single entry point for business logic:
  ingest_pdf()  →  extract → chunk → embed → index → persist
  search()      →  embed query → FAISS search → format results
  delete()      →  remove from index + disk
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List, Optional

from app.embeddings import encode_query, encode_texts
from app.pdf_processor import TextChunk, process_pdf
from app.vector_store import SearchResult, get_store

logger = logging.getLogger(__name__)


# ─── Ingestion ────────────────────────────────────────────────────────────────

def ingest_pdf(file_path: str | Path) -> dict:
    """
    Full ingestion pipeline for a single PDF.

    Steps:
      1. Generate a unique doc_id (UUID4).
      2. Extract text and chunk it (pdf_processor).
      3. Encode all chunk texts via HuggingFace (embeddings).
      4. Add vectors + metadata to FAISS (vector_store).

    Returns a summary dict suitable for returning from the API.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    doc_id = str(uuid.uuid4())
    logger.info("Starting ingestion for '%s' → doc_id=%s", file_path.name, doc_id)

    # 1. Extract text + build chunks
    chunks, metadata = process_pdf(file_path, doc_id)

    if not chunks:
        raise ValueError(f"No content could be extracted from '{file_path.name}'.")

    # 2. Embed all chunks
    texts      = [c.text for c in chunks]
    embeddings = encode_texts(texts, batch_size=32, normalize=True)

    # 3. Index
    store = get_store()
    store.add_document(chunks, embeddings, metadata)

    result = {
        "doc_id":      doc_id,
        "filename":    metadata.filename,
        "page_count":  metadata.page_count,
        "chunk_count": metadata.chunk_count,
        "total_chars": metadata.total_chars,
        "sha256":      metadata.sha256,
        "title":       metadata.title,
        "author":      metadata.author,
        "message":     f"Successfully indexed {metadata.chunk_count} chunks.",
    }
    logger.info("Ingestion complete: %s", result)
    return result


# ─── Search ───────────────────────────────────────────────────────────────────

def search_documents(
    query: str,
    top_k: int = 5,
    doc_ids: Optional[List[str]] = None,
) -> dict:
    """
    Semantic search across indexed documents.

    Args:
        query:   Natural-language query string.
        top_k:   Number of results to return.
        doc_ids: Optional list of doc_ids to scope the search.

    Returns dict with 'query', 'results', 'total_results'.
    """
    if not query.strip():
        raise ValueError("Query must be a non-empty string.")

    query_vec = encode_query(query.strip(), normalize=True)
    store     = get_store()
    hits: List[SearchResult] = store.search(query_vec, k=top_k, doc_ids=doc_ids)

    return {
        "query":         query,
        "total_results": len(hits),
        "results":       [h.to_dict() for h in hits],
    }


# ─── Document management ──────────────────────────────────────────────────────

def list_documents() -> dict:
    store = get_store()
    docs  = store.list_documents()
    return {
        "total": len(docs),
        "documents": [d.to_dict() for d in docs],
    }


def get_document_info(doc_id: str) -> Optional[dict]:
    store = get_store()
    meta  = store.get_document(doc_id)
    return meta.to_dict() if meta else None


def delete_document(doc_id: str, remove_file: bool = False) -> dict:
    """
    Remove a document from the vector index.
    Optionally also removes the original file from disk.
    """
    store   = get_store()
    meta    = store.get_document(doc_id)
    if meta is None:
        return {"success": False, "message": f"Document '{doc_id}' not found."}

    filename = meta.filename
    file_path = Path(meta.file_path)

    deleted = store.delete_document(doc_id)

    if remove_file and file_path.exists():
        try:
            file_path.unlink()
            logger.info("Deleted file: %s", file_path)
        except OSError as exc:
            logger.warning("Could not delete file %s: %s", file_path, exc)

    return {
        "success":  deleted,
        "doc_id":   doc_id,
        "filename": filename,
        "message":  f"Document '{filename}' removed from index.",
    }


def store_stats() -> dict:
    return get_store().stats()
