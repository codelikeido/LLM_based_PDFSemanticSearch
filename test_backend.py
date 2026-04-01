"""
tests/test_backend.py - Unit + integration tests for the PDF Search backend.

Run:
  python -m pytest tests/ -v
  python -m pytest tests/ -v --tb=short -q   # quiet mode
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# ── Ensure project root is on sys.path ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import EMBEDDING_DIM


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_dummy_pdf_bytes() -> bytes:
    """Return a minimal valid PDF with one page of extractable text."""
    content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello World PDF) Tj ET\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000360 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n441\n%%EOF\n"
    )
    return content


# ─────────────────────────────────────────────────────────────────────────────
# 1. PDF Processor Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestChunking(unittest.TestCase):
    """Test text chunking logic without touching files or models."""

    def _make_pages(self, texts):
        from app.pdf_processor import PageContent
        return [PageContent(page_number=i + 1, text=t) for i, t in enumerate(texts)]

    def test_single_page_produces_chunks(self):
        from app.pdf_processor import chunk_pages
        text  = ("This is a sentence. " * 50).strip()
        pages = self._make_pages([text])
        chunks = chunk_pages(pages, doc_id="test_doc", chunk_size=200, overlap=50)
        self.assertGreater(len(chunks), 0)

    def test_chunk_ids_are_unique(self):
        from app.pdf_processor import chunk_pages
        text   = ("Another sentence here. " * 100).strip()
        pages  = self._make_pages([text])
        chunks = chunk_pages(pages, doc_id="doc1", chunk_size=150, overlap=30)
        ids    = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)), "Chunk IDs must be unique")

    def test_chunk_text_is_non_empty(self):
        from app.pdf_processor import chunk_pages
        text   = "Word " * 300
        pages  = self._make_pages([text])
        chunks = chunk_pages(pages, doc_id="doc2", chunk_size=100, overlap=20)
        for chunk in chunks:
            self.assertTrue(chunk.text.strip(), "No empty chunks allowed")

    def test_overlap_creates_more_chunks_than_no_overlap(self):
        from app.pdf_processor import chunk_pages
        text   = "Sentence number here. " * 60
        pages  = self._make_pages([text])
        with_overlap    = chunk_pages(pages, "d1", chunk_size=100, overlap=40)
        without_overlap = chunk_pages(pages, "d2", chunk_size=100, overlap=0)
        self.assertGreaterEqual(len(with_overlap), len(without_overlap))

    def test_page_numbers_preserved(self):
        from app.pdf_processor import chunk_pages
        pages  = self._make_pages(["Page one text. " * 30, "Page two text. " * 30])
        chunks = chunk_pages(pages, "doc3", chunk_size=200, overlap=0)
        pnums  = {c.page_number for c in chunks}
        self.assertTrue(pnums.issubset({1, 2}))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Embedding Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddings(unittest.TestCase):

    def test_encode_returns_correct_shape(self):
        from app.embeddings import encode_texts, get_embedding_dim
        texts = ["Hello world", "Second sentence", "Third one"]
        vecs  = encode_texts(texts)
        self.assertEqual(vecs.shape, (3, get_embedding_dim()))
        self.assertEqual(vecs.dtype, np.float32)

    def test_encode_empty_list(self):
        from app.embeddings import encode_texts, get_embedding_dim
        vecs = encode_texts([])
        self.assertEqual(vecs.shape, (0, get_embedding_dim()))

    def test_query_encoding(self):
        from app.embeddings import encode_query, get_embedding_dim
        vec = encode_query("What is machine learning?")
        self.assertEqual(vec.shape, (1, get_embedding_dim()))

    def test_normalised_embeddings_have_unit_norm(self):
        from app.embeddings import encode_texts
        vecs  = encode_texts(["Normalisation test sentence"], normalize=True)
        norm  = float(np.linalg.norm(vecs[0]))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_same_text_same_embedding(self):
        """Embeddings must be deterministic for the same input."""
        from app.embeddings import encode_texts
        text = "Deterministic embedding test"
        v1   = encode_texts([text])
        v2   = encode_texts([text])
        np.testing.assert_array_almost_equal(v1, v2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Vector Store Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorStore(unittest.TestCase):

    def setUp(self):
        """Each test gets a fresh in-memory store backed by a temp dir."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        # Patch INDEX_DIR to an isolated temp directory
        self.patcher = patch("app.vector_store.INDEX_DIR", Path(self.tmp_dir.name))
        self.patcher.start()
        patch("app.vector_store._INDEX_FILE",
              Path(self.tmp_dir.name) / "faiss.index").start()
        patch("app.vector_store._META_FILE",
              Path(self.tmp_dir.name) / "metadata.pkl").start()

        from app.vector_store import VectorStore
        self.store = VectorStore()

    def tearDown(self):
        self.patcher.stop()
        self.tmp_dir.cleanup()

    def _make_doc(self, doc_id="doc_001", n_chunks=5):
        from app.pdf_processor import DocumentMetadata, TextChunk
        from app.embeddings import get_embedding_dim
        dim    = get_embedding_dim()
        chunks = [
            TextChunk(
                chunk_id    = f"{doc_id}_chunk_{i:05d}",
                doc_id      = doc_id,
                text        = f"Sample text for chunk {i} of {doc_id}.",
                page_number = 1,
                chunk_index = i,
                char_start  = i * 50,
                char_end    = i * 50 + 50,
            )
            for i in range(n_chunks)
        ]
        rng   = np.random.default_rng(42)
        vecs  = rng.standard_normal((n_chunks, dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= norms
        meta = DocumentMetadata(
            doc_id      = doc_id,
            filename    = f"{doc_id}.pdf",
            file_path   = f"/tmp/{doc_id}.pdf",
            page_count  = 2,
            total_chars = n_chunks * 50,
            chunk_count = n_chunks,
            sha256      = "deadbeef",
        )
        return chunks, vecs, meta

    def test_add_and_list(self):
        chunks, vecs, meta = self._make_doc()
        self.store.add_document(chunks, vecs, meta)
        docs = self.store.list_documents()
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].doc_id, "doc_001")

    def test_search_returns_results(self):
        from app.embeddings import get_embedding_dim
        chunks, vecs, meta = self._make_doc()
        self.store.add_document(chunks, vecs, meta)
        # Search with the first chunk's vector
        query = vecs[0:1]
        results = self.store.search(query, k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].rank, 1)

    def test_search_empty_store(self):
        from app.embeddings import get_embedding_dim
        query   = np.zeros((1, get_embedding_dim()), dtype=np.float32)
        results = self.store.search(query, k=5)
        self.assertEqual(results, [])

    def test_delete_document(self):
        chunks, vecs, meta = self._make_doc()
        self.store.add_document(chunks, vecs, meta)
        ok = self.store.delete_document("doc_001")
        self.assertTrue(ok)
        self.assertEqual(len(self.store.list_documents()), 0)

    def test_delete_nonexistent(self):
        ok = self.store.delete_document("does_not_exist")
        self.assertFalse(ok)

    def test_filter_by_doc_ids(self):
        from app.embeddings import get_embedding_dim
        chunks1, vecs1, meta1 = self._make_doc("doc_a", n_chunks=5)
        chunks2, vecs2, meta2 = self._make_doc("doc_b", n_chunks=5)
        self.store.add_document(chunks1, vecs1, meta1)
        self.store.add_document(chunks2, vecs2, meta2)

        query   = vecs1[0:1]
        results = self.store.search(query, k=10, doc_ids=["doc_a"])
        for r in results:
            self.assertEqual(r.chunk.doc_id, "doc_a")

    def test_stats(self):
        chunks, vecs, meta = self._make_doc(n_chunks=3)
        self.store.add_document(chunks, vecs, meta)
        stats = self.store.stats()
        self.assertEqual(stats["total_documents"], 1)
        self.assertEqual(stats["total_vectors"], 3)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Flask API Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIRoutes(unittest.TestCase):

    def setUp(self):
        """Spin up a test Flask client with mocked document_service."""
        self.tmp_dir = tempfile.TemporaryDirectory()

        # Patch every service function so tests don't touch disk or models
        self.patches = [
            patch("app.routes.ingest_pdf"),
            patch("app.routes.search_documents"),
            patch("app.routes.list_documents"),
            patch("app.routes.get_document_info"),
            patch("app.routes.delete_document"),
            patch("app.routes.store_stats"),
        ]
        self.mocks = {p.attribute: p.start() for p in self.patches}

        from app.factory import create_app
        self.app    = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp_dir.cleanup()

    # health

    def test_health_endpoint(self):
        self.mocks["store_stats"].return_value = {
            "total_vectors": 0, "total_documents": 0
        }
        resp = self.client.get("/api/v1/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")

    # list documents

    def test_list_documents(self):
        self.mocks["list_documents"].return_value = {"total": 0, "documents": []}
        resp = self.client.get("/api/v1/documents")
        self.assertEqual(resp.status_code, 200)

    # get single document

    def test_get_existing_document(self):
        self.mocks["get_document_info"].return_value = {
            "doc_id": "abc", "filename": "test.pdf"
        }
        resp = self.client.get("/api/v1/documents/abc")
        self.assertEqual(resp.status_code, 200)

    def test_get_missing_document(self):
        self.mocks["get_document_info"].return_value = None
        resp = self.client.get("/api/v1/documents/missing")
        self.assertEqual(resp.status_code, 404)

    # upload

    def test_upload_no_file(self):
        resp = self.client.post("/api/v1/documents/upload")
        self.assertEqual(resp.status_code, 400)

    def test_upload_wrong_extension(self):
        data = {"file": (io.BytesIO(b"data"), "doc.txt")}
        resp = self.client.post(
            "/api/v1/documents/upload",
            data=data,
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)

    def test_upload_valid_pdf(self):
        self.mocks["ingest_pdf"].return_value = {
            "doc_id":      "uuid-123",
            "filename":    "sample.pdf",
            "page_count":  1,
            "chunk_count": 3,
            "total_chars": 300,
            "sha256":      "abc",
            "message":     "Indexed 3 chunks.",
        }
        pdf_bytes = _make_dummy_pdf_bytes()
        data = {"file": (io.BytesIO(pdf_bytes), "sample.pdf")}
        resp = self.client.post(
            "/api/v1/documents/upload",
            data=data,
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 201)
        result = resp.get_json()
        self.assertEqual(result["doc_id"], "uuid-123")

    # search

    def test_search_missing_query(self):
        resp = self.client.post(
            "/api/v1/search",
            json={},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_search_empty_query(self):
        resp = self.client.post(
            "/api/v1/search",
            json={"query": "   "},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_search_valid(self):
        self.mocks["search_documents"].return_value = {
            "query":         "machine learning",
            "total_results": 2,
            "results": [
                {
                    "rank": 1, "score": 0.91,
                    "chunk_id": "doc_chunk_00001", "doc_id": "uuid",
                    "page_number": 1, "chunk_index": 1,
                    "text": "ML is a subset of AI.",
                },
                {
                    "rank": 2, "score": 0.82,
                    "chunk_id": "doc_chunk_00002", "doc_id": "uuid",
                    "page_number": 2, "chunk_index": 2,
                    "text": "Deep learning uses neural networks.",
                },
            ],
        }
        resp = self.client.post(
            "/api/v1/search",
            json={"query": "machine learning", "top_k": 2},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["total_results"], 2)
        self.assertEqual(data["results"][0]["rank"], 1)

    def test_search_invalid_top_k(self):
        resp = self.client.post(
            "/api/v1/search",
            json={"query": "test", "top_k": 999},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    # delete

    def test_delete_document(self):
        self.mocks["delete_document"].return_value = {
            "success": True, "doc_id": "abc", "message": "Removed."
        }
        resp = self.client.delete("/api/v1/documents/abc")
        self.assertEqual(resp.status_code, 200)

    def test_delete_missing_document(self):
        self.mocks["delete_document"].return_value = {
            "success": False, "message": "Not found."
        }
        resp = self.client.delete("/api/v1/documents/nope")
        self.assertEqual(resp.status_code, 404)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
