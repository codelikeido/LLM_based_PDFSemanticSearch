"""
config.py - Central configuration for the PDF Semantic Search Backend.
All tunable knobs live here; override via environment variables.
"""
import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
UPLOAD_DIR  = Path(os.getenv("UPLOAD_DIR",  BASE_DIR / "uploads"))
INDEX_DIR   = Path(os.getenv("INDEX_DIR",   BASE_DIR / "indexes"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True,  exist_ok=True)

# ─── Embedding model (HuggingFace / sentence-transformers) ────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, fast & accurate
)
EMBEDDING_DIM   = int(os.getenv("EMBEDDING_DIM", 384))

# ─── Text chunking ────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    500))   # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))   # overlap between chunks

# ─── FAISS ────────────────────────────────────────────────────────────────────
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "Flat")   # Flat | IVF | HNSW
FAISS_NLIST      = int(os.getenv("FAISS_NLIST", 100))       # for IVF indexes
TOP_K_DEFAULT    = int(os.getenv("TOP_K_DEFAULT", 5))

# ─── Flask ────────────────────────────────────────────────────────────────────
SECRET_KEY     = os.getenv("SECRET_KEY", "change-me-in-production")
MAX_CONTENT_MB = int(os.getenv("MAX_CONTENT_MB", 50))
ALLOWED_EXTENSIONS = {"pdf"}

# ─── Misc ─────────────────────────────────────────────────────────────────────
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
