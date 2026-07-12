"""
Run this once (and again whenever your source documents change) to
(re)build the Neo4j knowledge graph:

    python build_graph.py

Requires NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD to be set in .env,
pointing at a running Neo4j instance (Aura free tier works fine).
"""

from app.graph import build_graph
from app.config import RAG_DOCS_DIR

if __name__ == "__main__":
    build_graph(docs_dir=RAG_DOCS_DIR)
