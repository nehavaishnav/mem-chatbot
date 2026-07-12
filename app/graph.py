"""
Knowledge Graph Layer
----------------------
Extracts entities and relationships from the same source documents used
for RAG, and stores them as a graph in Neo4j. This lets the chatbot
answer *relational* questions ("what plan includes X", "who approves Y")
that plain text-chunk retrieval handles poorly, because the connection
between two facts is explicit graph structure rather than something that
has to be re-inferred from nearby words every time.

Extraction uses Gemini itself: given a chunk of text, ask it to return a
list of (subject, relation, object) triples as JSON. This is much less
work than a dedicated NER/relation-extraction pipeline and is good
enough for a demo-scale document set.
"""

import json
import os

from google import genai
from google.genai import types
from neo4j import GraphDatabase

from app.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    RAG_DOCS_DIR,
)

_client = genai.Client(api_key=GEMINI_API_KEY)

EXTRACTION_PROMPT = """Extract factual relationships from the text below as (subject, relation, object) triples.

Rules:
- Keep subject/object as short noun phrases (2-4 words max), relation as a short verb phrase (snake_case, e.g. "costs", "includes", "requires").
- Only extract clear, explicit relationships stated in the text. Do not infer or guess.
- Return ONLY a JSON array of objects with keys "subject", "relation", "object". No other text, no markdown fences.

Text:
{text}
"""


def _extract_triples(text: str) -> list:
    response = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=EXTRACTION_PROMPT.format(text=text),
        config=types.GenerateContentConfig(
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = (response.text or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        triples = json.loads(raw)
        return [t for t in triples if all(k in t for k in ("subject", "relation", "object"))]
    except (json.JSONDecodeError, TypeError):
        return []


def _get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def build_graph(docs_dir: str = RAG_DOCS_DIR) -> int:
    """Reads every .txt file, extracts triples, and writes them into Neo4j. Returns triple count."""
    driver = _get_driver()
    total = 0
    try:
        with driver.session() as session:
            # Start clean so re-running ingestion doesn't duplicate relationships
            session.run("MATCH (n) DETACH DELETE n")

            for fname in sorted(os.listdir(docs_dir)):
                if not fname.endswith(".txt"):
                    continue
                with open(os.path.join(docs_dir, fname), "r", encoding="utf-8") as f:
                    text = f.read()

                for para in [p for p in text.split("\n\n") if p.strip()]:
                    triples = _extract_triples(para)
                    for t in triples:
                        session.run(
                            """
                            MERGE (a:Entity {name: $subject})
                            MERGE (b:Entity {name: $object})
                            MERGE (a)-[r:RELATION {type: $relation}]->(b)
                            SET r.source = $source
                            """,
                            subject=t["subject"],
                            object=t["object"],
                            relation=t["relation"],
                            source=fname,
                        )
                        total += 1
    finally:
        driver.close()

    print(f"Wrote {total} relationships into Neo4j from {docs_dir}")
    return total


def query_graph(question: str, max_results: int = 8) -> list:
    """
    Very simple keyword-based graph lookup: pull every relationship whose
    subject or object name appears (as a substring) in the question, so we
    don't need a full NL-to-Cypher layer for a demo-scale graph.
    """
    driver = _get_driver()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (a:Entity)-[r:RELATION]->(b:Entity)
                RETURN a.name AS subject, r.type AS relation, b.name AS object, r.source AS source
                LIMIT 500
                """
            )
            rows = [dict(record) for record in result]
    finally:
        driver.close()

    q_lower = question.lower()
    matches = [
        row for row in rows
        if row["subject"].lower() in q_lower or row["object"].lower() in q_lower
    ]
    return matches[:max_results]


def format_graph_context(matches: list) -> str:
    if not matches:
        return "No relevant relationships found in the knowledge graph."
    lines = [f"{m['subject']} --{m['relation']}--> {m['object']} [source: {m['source']}]" for m in matches]
    return "\n".join(lines)
