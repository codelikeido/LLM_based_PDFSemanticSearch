# LLM_based_PDFSemanticSearch
# PDF Semantic Search Backend

A production-ready backend that enables **semantic search over PDF documents** using **HuggingFace embeddings + FAISS vector database**.

Upload PDFs → Extract text → Generate embeddings → Perform natural language search.

---

## Features

*  Upload and index PDF documents
*  Semantic (meaning-based) search
*  Fast vector search using FAISS
*  HuggingFace `sentence-transformers` embeddings
*  Intelligent text chunking with overlap
*  Persistent storage (survives server restart)
*  Fully tested backend (unit + integration tests)

##  Architecture

PDF → Text Extraction → Chunking → Embeddings → FAISS → Search Results

### Pipeline:

1. PDF uploaded via API
2. Text extracted using `pdfplumber` / `pypdf`
3. Split into chunks (with overlap)
4. Converted into embeddings (HuggingFace)
5. Stored in FAISS index
6. Queried using semantic similarity

## Project Structure


pdf_search_backend/
│
├── app/
│   ├── config.py
│   ├── document_service.py
│   ├── embeddings.py
│   ├── factory.py
│   ├── pdf_processor.py
│   ├── routes.py
│   ├── vector_store.py
│
├── tests/
│   └── test_backend.py
│
├── uploads/
├── indexes/
├── main.py
├── requirements.txt
└── README.md
```




 it helps a lot!
