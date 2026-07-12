"""
Evaluation Framework
----------------------
Runs a small test set of questions through the full chat pipeline and
scores each answer on three axes, using Gemini itself as the judge:

  - Context relevance: did retrieval (RAG/graph/tool) actually pull in
    material relevant to the question?
  - Answer correctness: does the reply match the expected answer?
  - Faithfulness: is the reply actually grounded in the retrieved
    context, rather than the model making things up?

Run it with:

    python eval.py

Edit TEST_CASES below to add your own questions once you swap in your
own knowledge base.
"""

import json
import statistics

from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app import orchestrator

_client = genai.Client(api_key=GEMINI_API_KEY)

# Each case: a question + the key fact(s) the answer should contain.
# expected_route is optional — set it when you want to also check routing accuracy.
TEST_CASES = [
    {
        "question": "How many vacation days do full-time employees get per year?",
        "expected_answer": "18 days of paid time off per year, plus 10 public holidays",
        "expected_route": "rag",
    },
    {
        "question": "What's the price of the Pro plan?",
        "expected_answer": "$49 per month",
        "expected_route": "rag",
    },
    {
        "question": "How many days can I carry over unused vacation?",
        "expected_answer": "up to 5 days can be carried over",
        "expected_route": "rag",
    },
    {
        "question": "What's the current weather in Tokyo?",
        "expected_answer": "a live temperature and wind speed reading for Tokyo",
        "expected_route": "tool",
    },
]

JUDGE_PROMPT = """You are grading a chatbot's answer. Respond with ONLY a JSON object, no other text.

Question: {question}
Expected answer should contain: {expected_answer}
Retrieved context given to the chatbot: {context}
Chatbot's actual reply: {reply}

Score each from 0.0 to 1.0:
- "context_relevance": did the retrieved context actually contain information relevant to the question?
- "answer_correctness": does the chatbot's reply match/contain the expected answer?
- "faithfulness": is the reply supported by the retrieved context (not made up)?

Respond exactly as: {{"context_relevance": 0.0-1.0, "answer_correctness": 0.0-1.0, "faithfulness": 0.0-1.0, "notes": "one short sentence"}}
"""


def _context_summary(result: dict) -> str:
    parts = []
    if result.get("rag_chunks"):
        parts.append("RAG: " + " | ".join(c["text"][:150] for c in result["rag_chunks"]))
    if result.get("graph_matches"):
        parts.append("Graph: " + " | ".join(f"{m['subject']}-{m['relation']}-{m['object']}" for m in result["graph_matches"]))
    if result.get("tool_result"):
        parts.append("Tool: " + result["tool_result"])
    return "\n".join(parts) if parts else "(no context retrieved)"


def _judge(question: str, expected_answer: str, context: str, reply: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question, expected_answer=expected_answer, context=context, reply=reply
    )
    response = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=512,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = (response.text or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"context_relevance": 0.0, "answer_correctness": 0.0, "faithfulness": 0.0, "notes": "judge parse error"}


def run_eval():
    rows = []
    for case in TEST_CASES:
        result = orchestrator.run_chat("eval_user", case["question"])
        context = _context_summary(result)
        scores = _judge(case["question"], case["expected_answer"], context, result["reply"])

        route_ok = (
            "n/a" if "expected_route" not in case
            else ("✓" if result.get("route") == case["expected_route"] else f"✗ (got {result.get('route')})")
        )

        rows.append({
            "question": case["question"],
            "route": route_ok,
            **scores,
        })

    print(f"{'Question':<55} {'Route':<12} {'Relevance':<10} {'Correct':<10} {'Faithful':<10}")
    print("-" * 100)
    for r in rows:
        print(f"{r['question'][:53]:<55} {r['route']:<12} {r['context_relevance']:<10} {r['answer_correctness']:<10} {r['faithfulness']:<10}")

    for metric in ("context_relevance", "answer_correctness", "faithfulness"):
        avg = statistics.mean(r[metric] for r in rows)
        print(f"\nAverage {metric}: {avg:.2f}")


if __name__ == "__main__":
    run_eval()
