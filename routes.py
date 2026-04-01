"""
routes.py - Flask REST API for the PDF Semantic Search backend.

Endpoints:
  POST   /api/v1/documents/upload    Upload & index a PDF
  GET    /api/v1/documents           List all indexed documents
  GET    /api/v1/documents/<doc_id>  Get document metadata
  DELETE /api/v1/documents/<doc_id>  Remove document from index
  POST   /api/v1/search              Semantic search
  GET    /api/v1/health              Health check + store stats
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from flask import Blueprint, jsonify, request, current_app

from app.config import ALLOWED_EXTENSIONS, UPLOAD_DIR
from app.document_service import (
    delete_document,
    get_document_info,
    ingest_pdf,
    list_documents,
    search_documents,
    store_stats,
)

logger  = logging.getLogger(__name__)
api_bp  = Blueprint("api", __name__, url_prefix="/api/v1")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _error(message: str, status: int = 400) -> tuple:
    return jsonify({"error": message}), status


def _ok(payload: dict, status: int = 200) -> tuple:
    return jsonify(payload), status


# ─── Health ───────────────────────────────────────────────────────────────────

@api_bp.route("/health", methods=["GET"])
def health():
    """
    GET /api/v1/health
    Returns service status and vector store statistics.
    """
    stats = store_stats()
    return _ok({
        "status": "ok",
        "store":  stats,
    })


# ─── Documents ────────────────────────────────────────────────────────────────

@api_bp.route("/documents/upload", methods=["POST"])
def upload_document():
    """
    POST /api/v1/documents/upload
    Multipart form field: file (PDF)

    Response 202 (accepted — ingestion runs synchronously here):
    {
      "doc_id":      "uuid",
      "filename":    "report.pdf",
      "page_count":  12,
      "chunk_count": 48,
      "total_chars": 24000,
      "sha256":      "abc...",
      "message":     "Successfully indexed 48 chunks."
    }
    """
    if "file" not in request.files:
        return _error("No file field in request.")

    file = request.files["file"]
    if file.filename == "":
        return _error("No file selected.")

    if not _allowed(file.filename):
        return _error(f"Only PDF files are accepted. Got: '{file.filename}'")

    # Save to disk with a collision-safe name
    safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    dest_path = UPLOAD_DIR / safe_name

    try:
        file.save(str(dest_path))
        logger.info("Saved upload to %s", dest_path)
    except OSError as exc:
        return _error(f"Could not save file: {exc}", 500)

    # Run ingestion pipeline
    try:
        result = ingest_pdf(dest_path)
        return _ok(result, 201)
    except ValueError as exc:
        dest_path.unlink(missing_ok=True)
        return _error(str(exc), 422)
    except Exception as exc:
        logger.exception("Ingestion failed for %s", dest_path)
        dest_path.unlink(missing_ok=True)
        return _error(f"Ingestion error: {exc}", 500)


@api_bp.route("/documents", methods=["GET"])
def list_docs():
    """
    GET /api/v1/documents
    Optional query params:
      page  (int, default 1)
      limit (int, default 20, max 100)

    Response:
    {
      "total": 5,
      "page":  1,
      "limit": 20,
      "documents": [ {...}, ... ]
    }
    """
    page  = max(1, request.args.get("page",  1,  type=int))
    limit = min(100, max(1, request.args.get("limit", 20, type=int)))

    data  = list_documents()
    docs  = data["documents"]

    start = (page - 1) * limit
    end   = start + limit
    page_docs = docs[start:end]

    return _ok({
        "total":     data["total"],
        "page":      page,
        "limit":     limit,
        "documents": page_docs,
    })


@api_bp.route("/documents/<doc_id>", methods=["GET"])
def get_doc(doc_id: str):
    """
    GET /api/v1/documents/<doc_id>
    Returns metadata for a single document.
    """
    info = get_document_info(doc_id)
    if info is None:
        return _error(f"Document '{doc_id}' not found.", 404)
    return _ok(info)


@api_bp.route("/documents/<doc_id>", methods=["DELETE"])
def delete_doc(doc_id: str):
    """
    DELETE /api/v1/documents/<doc_id>
    Query param:
      remove_file=true   — also delete the PDF from disk (default: false)

    Response:
    { "success": true, "doc_id": "...", "message": "..." }
    """
    remove_file = request.args.get("remove_file", "false").lower() == "true"
    result      = delete_document(doc_id, remove_file=remove_file)
    status      = 200 if result["success"] else 404
    return jsonify(result), status


# ─── Search ───────────────────────────────────────────────────────────────────

@api_bp.route("/search", methods=["POST"])
def search():
    """
    POST /api/v1/search
    Body (JSON):
    {
      "query":   "What are the key findings?",   // required
      "top_k":   5,                               // optional, default 5
      "doc_ids": ["uuid1", "uuid2"]              // optional, scopes search
    }

    Response:
    {
      "query":         "What are the key findings?",
      "total_results": 5,
      "results": [
        {
          "rank":        1,
          "score":       0.8912,
          "chunk_id":    "uuid_chunk_00001",
          "doc_id":      "uuid",
          "page_number": 3,
          "chunk_index": 4,
          "text":        "The key findings indicate ..."
        },
        ...
      ]
    }
    """
    body = request.get_json(silent=True)
    if not body:
        return _error("Request body must be JSON.")

    query   = body.get("query", "").strip()
    top_k   = int(body.get("top_k", 5))
    doc_ids = body.get("doc_ids")

    if not query:
        return _error("'query' field is required and must be non-empty.")
    if top_k < 1 or top_k > 50:
        return _error("'top_k' must be between 1 and 50.")
    if doc_ids is not None and not isinstance(doc_ids, list):
        return _error("'doc_ids' must be a list of strings.")

    try:
        result = search_documents(query, top_k=top_k, doc_ids=doc_ids or None)
        return _ok(result)
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        logger.exception("Search failed for query: %s", query)
        return _error(f"Search error: {exc}", 500)


# ─── Error handlers ───────────────────────────────────────────────────────────

@api_bp.app_errorhandler(404)
def not_found(e):
    return _error("Endpoint not found.", 404)


@api_bp.app_errorhandler(405)
def method_not_allowed(e):
    return _error("Method not allowed.", 405)


@api_bp.app_errorhandler(413)
def too_large(e):
    max_mb = current_app.config.get("MAX_CONTENT_LENGTH", 0) // (1024 * 1024)
    return _error(f"File too large. Maximum size is {max_mb} MB.", 413)
