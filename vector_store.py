"""
vector_store.py - FAISS-backed vector store for semantic document search.

Architecture:
  ┌──────────────────────────────────────┐
  │  VectorStore                         │
  │  ├── faiss.Index  (in-memory)        │
  │  ├── chunk_map    (id → TextChunk)   │
  │  └── doc_registry (doc_id → meta)   │
  └──────────────────────────────────────┘

The store is persisted to disk after each mutation (add/delete) so
the server can restart without losing indexed data.

Supported FAISS index types (set via config.FAISS_INDEX_TYPE):
  • "Flat"  — exact L2/IP search; best for <100k vectors
  • "IVF"   — approximate; faster for >100k vectors (requires training)
  • "HNSW"  — graph-based approximate; very fast query, no training needed
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.config import (
    EMBEDDING_DIM,
    FAISS_INDEX_TYPE,
    FAISS_NLIST,
    INDEX_DIR,
    TOP_K_DEFAULT,
)
from app.pdf_processor import DocumentMetadata, TextChunk

logger = logging.getLogger(__name__)

# ─── FAISS import (with graceful fallback) ───────────────────────────────────

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning(
        "faiss-cpu not installed — using numpy brute-force search. "
        "Install with: pip install faiss-cpu"
    )


# ─── Search result ────────────────────────────────────────────────────────────

class SearchResult:
    """A single semantic search hit."""

    def __init__(self, chunk: TextChunk, score: float, rank: int):
        self.chunk = chunk
        self.score = score     # higher = more similar (cosine or negative L2)
        self.rank  = rank

    def to_dict(self) -> dict:
        return {
            "rank":        self.rank,
            "score":       round(float(self.score), 4),
            "chunk_id":    self.chunk.chunk_id,
            "doc_id":      self.chunk.doc_id,
            "page_number": self.chunk.page_number,
            "chunk_index": self.chunk.chunk_index,
            "text":        self.chunk.text,
        }


# ─── Numpy brute-force fallback ───────────────────────────────────────────────

class _NumpyIndex:
    """Pure-numpy cosine similarity index — exact but slow for large corpora."""

    def __init__(self, dim: int):
        self.dim    = dim
        self._vecs: Optional[np.ndarray] = None  # shape (N, dim)
        self._ids:  List[int] = []               # internal integer IDs

    def add_with_ids(self, vecs: np.ndarray, ids: np.ndarray):
        assert vecs.shape[1] == self.dim
        if self._vecs is None:
            self._vecs = vecs.copy()
        else:
            self._vecs = np.vstack([self._vecs, vecs])
        self._ids.extend(ids.tolist())

    def search(self, query: np.ndarray, k: int):
        """Returns (distances, ids) matching FAISS interface."""
        if self._vecs is None or len(self._ids) == 0:
            return np.array([[-1.0] * k]), np.array([[-1] * k])

        # Cosine similarity (vectors should already be L2-normalised)
        sims = (self._vecs @ query.T).flatten()   # shape (N,)
        k    = min(k, len(sims))
        top  = np.argsort(sims)[::-1][:k]
        dists= sims[top]
        ids  = np.array([self._ids[i] for i in top])
        return dists.reshape(1, -1), ids.reshape(1, -1)

    def remove_ids(self, ids_to_remove: set):
        if self._vecs is None:
            return
        keep = [i for i, id_ in enumerate(self._ids) if id_ not in ids_to_remove]
        if keep:
            self._vecs = self._vecs[keep]
            self._ids  = [self._ids[i] for i in keep]
        else:
            self._vecs = None
            self._ids  = []

    @property
    def ntotal(self) -> int:
        return len(self._ids)


# ─── VectorStore ─────────────────────────────────────────────────────────────

_INDEX_FILE    = INDEX_DIR / "faiss.index"
_META_FILE     = INDEX_DIR / "metadata.pkl"


class VectorStore:
    """Thread-safe FAISS (or numpy fallback) vector store."""

    def __init__(self):
        self._lock            = threading.RLock()
        self._index           = None        # faiss.Index or _NumpyIndex
        self._chunk_map: Dict[int, TextChunk]        = {}   # int_id → chunk
        self._doc_registry: Dict[str, DocumentMetadata] = {}   # doc_id → meta
        self._next_id: int    = 0
        self._doc_to_ids: Dict[str, List[int]] = {}   # doc_id → [int_ids]
        self._load_or_init()

    # ── Construction ──────────────────────────────────────────────────────────

    def _build_faiss_index(self):
        dim = EMBEDDING_DIM
        if not FAISS_AVAILABLE:
            return _NumpyIndex(dim)

        if FAISS_INDEX_TYPE == "IVF":
            quantiser = faiss.IndexFlatIP(dim)
            index     = faiss.IndexIVFFlat(quantiser, dim, FAISS_NLIST,
                                            faiss.METRIC_INNER_PRODUCT)
        elif FAISS_INDEX_TYPE == "HNSW":
            index = faiss.IndexHNSWFlat(dim, 32)
            index.metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            # Default: exact inner-product (cosine on normalised vecs)
            index = faiss.IndexFlatIP(dim)

        # Wrap with IDMap so we can store our own integer IDs
        return faiss.IndexIDMap(index)

    def _load_or_init(self):
        if _INDEX_FILE.exists() and _META_FILE.exists():
            self._load()
        else:
            self._index = self._build_faiss_index()
            logger.info("Initialized new %s FAISS index (dim=%d)",
                        FAISS_INDEX_TYPE, EMBEDDING_DIM)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        """Persist index + metadata to disk (call inside self._lock)."""
        try:
            if FAISS_AVAILABLE and not isinstance(self._index, _NumpyIndex):
                faiss.write_index(self._index, str(_INDEX_FILE))
            else:
                with open(_INDEX_FILE.with_suffix(".npy.pkl"), "wb") as f:
                    pickle.dump(self._index, f)

            meta_payload = {
                "chunk_map":    {k: v.to_dict() for k, v in self._chunk_map.items()},
                "doc_registry": {k: v.to_dict() for k, v in self._doc_registry.items()},
                "next_id":      self._next_id,
                "doc_to_ids":   self._doc_to_ids,
            }
            with open(_META_FILE, "w") as f:
                json.dump(meta_payload, f)

            logger.debug("VectorStore persisted (%d vectors)", self._index.ntotal)
        except Exception as exc:
            logger.error("Failed to save VectorStore: %s", exc)

    def _load(self):
        """Load persisted index + metadata from disk."""
        try:
            if FAISS_AVAILABLE and _INDEX_FILE.exists():
                self._index = faiss.read_index(str(_INDEX_FILE))
            elif _INDEX_FILE.with_suffix(".npy.pkl").exists():
                with open(_INDEX_FILE.with_suffix(".npy.pkl"), "rb") as f:
                    self._index = pickle.load(f)
            else:
                self._index = self._build_faiss_index()

            with open(_META_FILE) as f:
                payload = json.load(f)

            # Reconstruct TextChunk objects
            self._chunk_map = {
                int(k): TextChunk(**v)
                for k, v in payload["chunk_map"].items()
            }
            self._doc_registry = {
                k: DocumentMetadata(**v)
                for k, v in payload["doc_registry"].items()
            }
            self._next_id    = payload["next_id"]
            self._doc_to_ids = {k: v for k, v in payload["doc_to_ids"].items()}

            logger.info(
                "VectorStore loaded: %d vectors, %d documents",
                self._index.ntotal, len(self._doc_registry),
            )
        except Exception as exc:
            logger.error("Failed to load VectorStore (%s) — reinitialising.", exc)
            self._index       = self._build_faiss_index()
            self._chunk_map   = {}
            self._doc_registry= {}
            self._next_id     = 0
            self._doc_to_ids  = {}

    # ── Public write API ──────────────────────────────────────────────────────

    def add_document(
        self,
        chunks: List[TextChunk],
        embeddings: np.ndarray,
        metadata: DocumentMetadata,
    ) -> str:
        """
        Index all chunks + embeddings for a document.
        Returns doc_id.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings length mismatch")

        with self._lock:
            doc_id = metadata.doc_id

            # Allocate integer IDs
            n        = len(chunks)
            int_ids  = list(range(self._next_id, self._next_id + n))
            self._next_id += n

            # Store chunk metadata
            for int_id, chunk in zip(int_ids, chunks):
                self._chunk_map[int_id] = chunk

            # Add to FAISS
            ids_arr = np.array(int_ids, dtype=np.int64)

            if FAISS_AVAILABLE and not isinstance(self._index, _NumpyIndex):
                # IVF index needs training before first use
                if (FAISS_INDEX_TYPE == "IVF"
                        and not self._index.index.is_trained
                        and embeddings.shape[0] >= FAISS_NLIST):
                    logger.info("Training IVF index with %d vectors", n)
                    self._index.index.train(embeddings)

                self._index.add_with_ids(embeddings, ids_arr)
            else:
                self._index.add_with_ids(embeddings, ids_arr)

            # Register document
            self._doc_registry[doc_id] = metadata
            self._doc_to_ids.setdefault(doc_id, []).extend(int_ids)

            self._save()
            logger.info("Indexed doc_id=%s (%d chunks)", doc_id, n)
            return doc_id

    def delete_document(self, doc_id: str) -> bool:
        """Remove all chunks belonging to a document."""
        with self._lock:
            if doc_id not in self._doc_registry:
                return False

            int_ids = set(self._doc_to_ids.pop(doc_id, []))

            if FAISS_AVAILABLE and not isinstance(self._index, _NumpyIndex):
                # faiss.IndexIDMap supports remove_ids
                id_selector = faiss.IDSelectorArray(
                    np.array(list(int_ids), dtype=np.int64)
                )
                self._index.remove_ids(id_selector)
            else:
                self._index.remove_ids(int_ids)

            for id_ in int_ids:
                self._chunk_map.pop(id_, None)

            del self._doc_registry[doc_id]
            self._save()
            logger.info("Deleted doc_id=%s", doc_id)
            return True

    # ── Public read API ───────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = TOP_K_DEFAULT,
        doc_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Semantic search.

        Args:
            query_embedding: shape (1, dim), L2-normalised.
            k:               number of results.
            doc_ids:         if provided, filter results to these documents.

        Returns:
            List of SearchResult sorted by score descending.
        """
        with self._lock:
            if self._index.ntotal == 0:
                return []

            fetch_k = k * 5 if doc_ids else k   # over-fetch when filtering
            scores_arr, ids_arr = self._index.search(query_embedding, fetch_k)

            results: List[SearchResult] = []
            for score, int_id in zip(scores_arr[0], ids_arr[0]):
                if int_id < 0:           # FAISS pads with -1 when < k results
                    continue
                chunk = self._chunk_map.get(int(int_id))
                if chunk is None:
                    continue
                if doc_ids and chunk.doc_id not in doc_ids:
                    continue
                results.append(SearchResult(chunk, float(score), rank=0))
                if len(results) >= k:
                    break

            for i, r in enumerate(results, start=1):
                r.rank = i

            return results

    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        with self._lock:
            return self._doc_registry.get(doc_id)

    def list_documents(self) -> List[DocumentMetadata]:
        with self._lock:
            return list(self._doc_registry.values())

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_vectors":   self._index.ntotal,
                "total_documents": len(self._doc_registry),
                "total_chunks":    len(self._chunk_map),
                "index_type":      FAISS_INDEX_TYPE,
                "embedding_dim":   EMBEDDING_DIM,
                "faiss_available": FAISS_AVAILABLE,
            }


# ─── Module-level singleton ───────────────────────────────────────────────────

_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
