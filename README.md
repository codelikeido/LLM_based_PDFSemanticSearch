# PDF Semantic Search — Backend

A production-ready REST API that lets you **upload PDF documents**, **extract their text**, **generate dense embeddings** with HuggingFace `sentence-transformers`, index them in a **FAISS** vector store, and perform **natural-language semantic search** across your document corpus.

---

## Architecture

```
PDF File
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  pdf_processor.py                                       │
│  ├── pdfplumber  (primary extraction — layout-aware)    │
│  ├── pypdf       (fallback extraction)                  │
│  └── Sliding-window chunker (size=500, overlap=100)     │
└─────────────────────────────────────────────────────────┘
  │  List[TextChunk]
  ▼
┌─────────────────────────────────────────────────────────┐
│  embeddings.py                                          │
│  └── sentence-transformers/all-MiniLM-L6-v2            │
│      → float32 np.ndarray  shape (N, 384)              │
└─────────────────────────────────────────────────────────┘
  │  np.ndarray
  ▼
┌─────────────────────────────────────────────────────────┐
│  vector_store.py                                        │
│  ├── FAISS IndexFlatIP (exact cosine via L2-norm)       │
│  ├── chunk_map    { int_id → TextChunk }                │
│  └── Persisted to indexes/ on every write               │
└─────────────────────────────────────────────────────────┘
  │  SearchResult[]
  ▼
┌─────────────────────────────────────────────────────────┐
│  routes.py  (Flask REST API)                            │
│  └── JSON responses                                     │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
pdf_search_backend/
├── app/
│   ├── __init__.py
│   ├── config.py           # all config + env-var overrides
│   ├── pdf_processor.py    # extraction + chunking
│   ├── embeddings.py       # HuggingFace encode()
│   ├── vector_store.py     # FAISS index + persistence
│   ├── document_service.py # pipeline orchestration
│   ├── routes.py           # Flask REST endpoints
│   └── factory.py          # Flask app factory
├── tests/
│   └── test_backend.py     # unit + integration tests
├── uploads/                # uploaded PDFs (auto-created)
├── indexes/                # FAISS index + metadata (auto-created)
├── main.py                 # entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU acceleration**: replace `faiss-cpu` with `faiss-gpu` and ensure CUDA is set up.

### 2. Run the server

```bash
python main.py
```

Or with gunicorn (production):

```bash
gunicorn -w 2 -b 0.0.0.0:5000 main:app
```

### 3. Upload a PDF

```bash
curl -X POST http://localhost:5000/api/v1/documents/upload \
  -F "file=@/path/to/document.pdf"
```

Response:
```json
{
  "doc_id": "3e4f...",
  "filename": "document.pdf",
  "page_count": 12,
  "chunk_count": 47,
  "total_chars": 23500,
  "sha256": "abc123...",
  "message": "Successfully indexed 47 chunks."
}
```

### 4. Search

```bash
curl -X POST http://localhost:5000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key conclusions?", "top_k": 5}'
```

Response:
```json
{
  "query": "What are the key conclusions?",
  "total_results": 5,
  "results": [
    {
      "rank": 1,
      "score": 0.8921,
      "chunk_id": "3e4f_chunk_00012",
      "doc_id": "3e4f...",
      "page_number": 8,
      "chunk_index": 12,
      "text": "The key conclusions of this study are..."
    }
  ]
}
```

---

## API Reference

| Method   | Endpoint                          | Description                        |
|----------|-----------------------------------|------------------------------------|
| `GET`    | `/api/v1/health`                  | Health check + index stats         |
| `POST`   | `/api/v1/documents/upload`        | Upload & index a PDF               |
| `GET`    | `/api/v1/documents`               | List all indexed documents         |
| `GET`    | `/api/v1/documents/<doc_id>`      | Get document metadata              |
| `DELETE` | `/api/v1/documents/<doc_id>`      | Remove document from index         |
| `POST`   | `/api/v1/search`                  | Semantic search                    |

### POST /api/v1/search — Body

```json
{
  "query":   "string (required)",
  "top_k":   5,
  "doc_ids": ["uuid1", "uuid2"]
}
```

`doc_ids` scopes the search to specific documents (optional).

### DELETE /api/v1/documents/<doc_id>

Query param `?remove_file=true` also deletes the PDF from disk.

---

## Configuration

All settings are in `app/config.py` and can be overridden via environment variables:

| Variable           | Default                                    | Description                         |
|--------------------|--------------------------------------------|-------------------------------------|
| `UPLOAD_DIR`       | `uploads/`                                 | Where PDFs are stored               |
| `INDEX_DIR`        | `indexes/`                                 | Where FAISS index is persisted      |
| `EMBEDDING_MODEL`  | `sentence-transformers/all-MiniLM-L6-v2`  | HuggingFace model name              |
| `EMBEDDING_DIM`    | `384`                                      | Output dimension of the model       |
| `CHUNK_SIZE`       | `500`                                      | Characters per text chunk           |
| `CHUNK_OVERLAP`    | `100`                                      | Overlap between consecutive chunks  |
| `FAISS_INDEX_TYPE` | `Flat`                                     | `Flat` / `IVF` / `HNSW`            |
| `TOP_K_DEFAULT`    | `5`                                        | Default search results              |
| `MAX_CONTENT_MB`   | `50`                                       | Max upload file size in MB          |
| `DEBUG`            | `true`                                     | Flask debug mode                    |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

The test suite includes:
- **Text chunking** — uniqueness, overlap, page-number preservation
- **Embeddings** — shape, dtype, L2-normalisation, determinism
- **VectorStore** — add, search, filter, delete, stats
- **Flask API** — all endpoints with mocked service layer

---

## Swapping the Embedding Model

To use a larger / domain-specific model:

```bash
export EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
export EMBEDDING_DIM=768
python main.py
```

> After changing the model, **re-index** all documents because embedding dimensions will change.

---

## Graceful Degradation

If `sentence-transformers` or `faiss-cpu` are not installed:
- **Embeddings** fall back to a deterministic random-projection stub (warns loudly).
- **Vector search** falls back to a NumPy brute-force cosine search.

This keeps the server functional for development without GPU or heavy ML dependencies.
