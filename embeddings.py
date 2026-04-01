"""
embeddings.py - Generate and manage text embeddings via HuggingFace
sentence-transformers.

Design decisions:
  • Singleton model loader — the model is expensive to load; we load
    it once and reuse it for the lifetime of the process.
  • Batch inference — encode() sends chunks in configurable batches to
    balance GPU/CPU memory vs. throughput.
  • Graceful degradation — if sentence-transformers is not installed,
    the module falls back to a deterministic random-projection stub so
    the rest of the stack keeps working (useful for CI / testing).
"""
from __future__ import annotations

import logging
import time
from typing import List

import numpy as np

from app.config import EMBEDDING_MODEL, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# ─── Model singleton ─────────────────────────────────────────────────────────

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        t0 = time.time()
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Model loaded in %.1fs", time.time() - t0)
    except ImportError:
        logger.warning(
            "sentence-transformers not installed — using stub embeddings. "
            "Install with: pip install sentence-transformers"
        )
        _model = _StubEmbedder(EMBEDDING_DIM)

    return _model


# ─── Stub embedder (fallback) ─────────────────────────────────────────────────

class _StubEmbedder:
    """
    Deterministic random-projection stub.
    Produces embeddings that are consistent for the same input text,
    which allows the FAISS index to behave correctly in tests.
    WARNING: Semantic similarity is meaningless with this stub.
    """
    def __init__(self, dim: int):
        self.dim = dim
        logger.warning("⚠️  Using STUB embedder — semantic search is non-functional.")

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        result = []
        for text in texts:
            rng = np.random.default_rng(seed=abs(hash(text)) % (2**32))
            vec = rng.standard_normal(self.dim).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            result.append(vec)
        return np.vstack(result)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


# ─── Public API ───────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the output dimension of the active model."""
    model = _load_model()
    if hasattr(model, "get_sentence_embedding_dimension"):
        return model.get_sentence_embedding_dimension()
    return EMBEDDING_DIM


def encode_texts(
    texts: List[str],
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode a list of strings → float32 numpy array of shape (N, dim).

    Args:
        texts:      Input strings.
        batch_size: Batch size for inference.
        normalize:  L2-normalise embeddings (recommended for cosine search).

    Returns:
        np.ndarray of shape (len(texts), embedding_dim), dtype float32.
    """
    if not texts:
        return np.empty((0, get_embedding_dim()), dtype=np.float32)

    model = _load_model()

    t0 = time.time()
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size          = batch_size,
        show_progress_bar   = len(texts) > 50,
        convert_to_numpy    = True,
        normalize_embeddings= normalize,
    )
    embeddings = embeddings.astype(np.float32)

    logger.info(
        "Encoded %d texts → shape %s in %.2fs",
        len(texts), embeddings.shape, time.time() - t0,
    )
    return embeddings


def encode_query(query: str, normalize: bool = True) -> np.ndarray:
    """
    Encode a single query string → float32 array of shape (1, dim).
    Convenience wrapper around encode_texts.
    """
    vec = encode_texts([query], normalize=normalize)
    return vec  # shape (1, dim)
