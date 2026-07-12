"""
Run this once (and again whenever your source documents change) to
(re)build the RAG index:

    python ingest.py
"""

from app.rag import build_index
from app.config import RAG_DOCS_DIR, RAG_INDEX_PATH

if __name__ == "__main__":
    build_index(docs_dir=RAG_DOCS_DIR, index_path=RAG_INDEX_PATH)
