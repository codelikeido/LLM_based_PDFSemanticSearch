"""
pdf_processor.py - Extract, clean, and chunk text from PDF files.

Strategy (in order):
  1. pdfplumber  — best for layout-aware extraction (tables, columns)
  2. pypdf       — fallback for simple text-layer PDFs
  3. OCR hint    — if both fail, surface a clear error so caller can
                   direct user to pre-OCR the document.
"""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pdfplumber
from pypdf import PdfReader

from app.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class PageContent:
    """Text extracted from a single PDF page."""
    page_number: int          # 1-based
    text: str
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)


@dataclass
class TextChunk:
    """A fixed-size window of text with provenance metadata."""
    chunk_id: str             # unique within a document
    doc_id: str               # references the parent document
    text: str
    page_number: int          # page where the chunk starts
    chunk_index: int          # sequential index within the document
    char_start: int           # character offset from document start
    char_end: int

    def to_dict(self) -> dict:
        return {
            "chunk_id":    self.chunk_id,
            "doc_id":      self.doc_id,
            "text":        self.text,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "char_start":  self.char_start,
            "char_end":    self.char_end,
        }


@dataclass
class DocumentMetadata:
    """High-level metadata for an ingested PDF."""
    doc_id: str
    filename: str
    file_path: str
    page_count: int
    total_chars: int
    chunk_count: int
    sha256: str
    author: Optional[str]     = None
    title: Optional[str]      = None
    creation_date: Optional[str] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _clean_text(raw: str) -> str:
    """Normalise unicode, collapse whitespace, remove control characters."""
    text = unicodedata.normalize("NFKC", raw)
    # remove non-printable control chars (keep newlines/tabs)
    text = "".join(c for c in text if unicodedata.category(c) != "Cc"
                   or c in "\n\t")
    # collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # collapse internal whitespace on the same line
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ─── Extraction ───────────────────────────────────────────────────────────────

def _extract_with_pdfplumber(path: Path) -> List[PageContent]:
    pages: List[PageContent] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text() or ""
                cleaned = _clean_text(raw)
                if cleaned:
                    pages.append(PageContent(page_number=i, text=cleaned))
    except Exception as exc:
        logger.warning("pdfplumber failed on %s: %s", path.name, exc)
    return pages


def _extract_with_pypdf(path: Path) -> List[PageContent]:
    pages: List[PageContent] = []
    try:
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = _clean_text(raw)
            if cleaned:
                pages.append(PageContent(page_number=i, text=cleaned))
    except Exception as exc:
        logger.warning("pypdf failed on %s: %s", path.name, exc)
    return pages


def extract_pages(path: Path) -> List[PageContent]:
    """
    Extract text page-by-page. Falls back from pdfplumber → pypdf.
    Raises ValueError if no text is found (likely scanned / image-only PDF).
    """
    pages = _extract_with_pdfplumber(path)
    if not pages:
        logger.info("Falling back to pypdf for %s", path.name)
        pages = _extract_with_pypdf(path)

    if not pages:
        raise ValueError(
            f"No extractable text found in '{path.name}'. "
            "The file may be a scanned/image-only PDF. "
            "Please run OCR (e.g. ocrmypdf) before uploading."
        )

    total_chars = sum(p.char_count for p in pages)
    logger.info("Extracted %d pages / %d chars from %s",
                len(pages), total_chars, path.name)
    return pages


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_pages(
    pages: List[PageContent],
    doc_id: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int    = CHUNK_OVERLAP,
) -> List[TextChunk]:
    """
    Sliding-window chunking over the concatenated document text.
    Each chunk preserves which page it started on.

    Algorithm:
      1. Join pages with markers so we can recover page numbers.
      2. Slide a window of `chunk_size` chars with `overlap` step-back.
      3. Try to break on sentence boundaries ('. ', '? ', '! ').
    """
    # Build a mapping: global_char_offset → page_number
    page_map: list[tuple[int, int]] = []   # (start_offset, page_number)
    full_text_parts: list[str] = []
    offset = 0
    for p in pages:
        page_map.append((offset, p.page_number))
        full_text_parts.append(p.text)
        offset += len(p.text) + 1          # +1 for the joining newline

    full_text = "\n".join(full_text_parts)

    def page_at(char_pos: int) -> int:
        """Return page number for a given character offset."""
        pg = 1
        for start, pnum in page_map:
            if start <= char_pos:
                pg = pnum
            else:
                break
        return pg

    def smart_end(text: str, start: int, end: int) -> int:
        """
        Extend/trim `end` to the nearest sentence boundary within a small
        lookahead window, to avoid cutting mid-sentence.
        """
        lookahead = min(end + 80, len(text))
        window = text[end:lookahead]
        for sep in (". ", "? ", "! ", "\n\n"):
            idx = window.find(sep)
            if idx != -1:
                return end + idx + len(sep)
        return end

    chunks: List[TextChunk] = []
    start = 0
    idx   = 0

    while start < len(full_text):
        raw_end = min(start + chunk_size, len(full_text))
        end     = smart_end(full_text, start, raw_end)

        chunk_text = full_text[start:end].strip()
        if chunk_text:
            chunk_id = f"{doc_id}_chunk_{idx:05d}"
            chunks.append(TextChunk(
                chunk_id    = chunk_id,
                doc_id      = doc_id,
                text        = chunk_text,
                page_number = page_at(start),
                chunk_index = idx,
                char_start  = start,
                char_end    = end,
            ))
            idx += 1

        step = max(1, chunk_size - overlap)
        start += step

    logger.info("Created %d chunks for doc_id=%s", len(chunks), doc_id)
    return chunks


# ─── Public API ───────────────────────────────────────────────────────────────

def process_pdf(path: Path, doc_id: str) -> tuple[List[TextChunk], DocumentMetadata]:
    """
    Full pipeline: extract → chunk → metadata.
    Returns (chunks, metadata).
    """
    file_path = Path(path)

    # Extract PDF metadata
    meta_extra: dict = {}
    try:
        reader = PdfReader(str(file_path))
        info   = reader.metadata or {}
        meta_extra = {
            "author":        info.get("/Author"),
            "title":         info.get("/Title"),
            "creation_date": info.get("/CreationDate"),
            "page_count":    len(reader.pages),
        }
    except Exception:
        meta_extra = {"page_count": 0}

    pages  = extract_pages(file_path)
    chunks = chunk_pages(pages, doc_id)

    total_chars = sum(p.char_count for p in pages)

    metadata = DocumentMetadata(
        doc_id     = doc_id,
        filename   = file_path.name,
        file_path  = str(file_path),
        page_count = meta_extra.get("page_count", len(pages)),
        total_chars= total_chars,
        chunk_count= len(chunks),
        sha256     = _sha256(file_path),
        author     = meta_extra.get("author"),
        title      = meta_extra.get("title"),
        creation_date = meta_extra.get("creation_date"),
    )

    return chunks, metadata
