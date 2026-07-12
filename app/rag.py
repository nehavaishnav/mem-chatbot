"""
Static Knowledge Layer (RAG)
----------------------------
This is the "simplest path" version of the RAG layer described in the
project brief:

    web/text data -> clean -> chunk -> vectorize -> store -> retrieve

For a minimal demo we use scikit-learn's TF-IDF vectorizer instead of
neural embeddings + FAISS/Chroma. This keeps the system dependency-light
(no GPU / large model downloads) while demonstrating the exact same
pipeline shape. Swapping TF-IDF for real embeddings later only means
replacing `_vectorize()` and the similarity search — the rest of the
architecture (chunk -> store -> retrieve -> feed to LLM) stays the same.
"""

import os
import pickle
import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import RAG_DOCS_DIR, RAG_INDEX_PATH, TOP_K_CHUNKS


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


@dataclass
class RagIndex:
    vectorizer: TfidfVectorizer
    matrix: object  # sparse TF-IDF matrix, one row per chunk
    chunks: list = field(default_factory=list)


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(text: str, source: str, max_words: int = 80, overlap: int = 15) -> list:
    """Simple sliding-window word chunker. Good enough for FAQ/policy-style docs."""
    words = text.split()
    chunks = []
    start = 0
    cid = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_text = " ".join(words[start:end])
        if chunk_text.strip():
            chunks.append(Chunk(text=chunk_text, source=source, chunk_id=cid))
            cid += 1
        if end == len(words):
            break
        start = end - overlap
    return chunks


def build_index(docs_dir: str = RAG_DOCS_DIR, index_path: str = RAG_INDEX_PATH) -> RagIndex:
    """Reads every .txt file in docs_dir, chunks it, and builds a TF-IDF index."""
    all_chunks = []
    for fname in sorted(os.listdir(docs_dir)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(docs_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        cleaned = _clean_text(raw)
        # Chunk paragraph-by-paragraph first, then by word-window within long paragraphs
        for para in [p for p in cleaned.split("\n\n") if p.strip()]:
            all_chunks.extend(_chunk_text(para, source=fname))

    if not all_chunks:
        raise ValueError(f"No .txt documents found in {docs_dir}")

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([c.text for c in all_chunks])

    index = RagIndex(vectorizer=vectorizer, matrix=matrix, chunks=all_chunks)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump(index, f)

    print(f"Indexed {len(all_chunks)} chunks from {docs_dir} -> {index_path}")
    return index


def load_index(index_path: str = RAG_INDEX_PATH) -> RagIndex:
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No index found at {index_path}. Run `python ingest.py` first."
        )
    with open(index_path, "rb") as f:
        return pickle.load(f)


def retrieve(query: str, index: RagIndex, top_k: int = TOP_K_CHUNKS) -> list:
    """Returns the top_k most relevant chunks for a query, each as a dict."""
    query_vec = index.vectorizer.transform([query])
    sims = cosine_similarity(query_vec, index.matrix)[0]
    ranked = sims.argsort()[::-1][:top_k]

    results = []
    for i in ranked:
        if sims[i] <= 0:
            continue
        c = index.chunks[i]
        results.append({"text": c.text, "source": c.source, "score": float(sims[i])})
    return results
