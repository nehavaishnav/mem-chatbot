"""
Dynamic Intelligence Layer — answer generation
------------------------------------------------
This module builds the final prompt and calls Gemini. It no longer
decides *which* context to pull in — that's now LangGraph's job
(see app/orchestrator.py). This module just knows how to render
whatever context it's handed (RAG chunks, knowledge-graph relationships,
a tool result) into a system prompt and generate the reply.
"""

import re

from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app import memory

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with access to a knowledge base, a knowledge graph, live tools, and long-term memory about the user.

Known facts about this user (long-term memory):
{profile}

Relevant knowledge base excerpts for this query:
{rag_context}

Relevant knowledge graph relationships for this query:
{graph_context}

Live tool result for this query (if any):
{tool_context}

Instructions:
- Prefer the knowledge base, graph, and tool results above over your own general knowledge when they're relevant.
- Use the known facts about the user to personalize your answer where it helps.
- If none of the provided context contains the answer, say so plainly rather than guessing.
- Be concise and direct.
"""


def _format_rag_context(chunks: list) -> str:
    if not chunks:
        return "No relevant excerpts found."
    return "\n\n".join(f"[source: {c['source']}] {c['text']}" for c in chunks)


def _format_graph_context(matches: list) -> str:
    if not matches:
        return "No relevant relationships found."
    return "\n".join(
        f"{m['subject']} --{m['relation']}--> {m['object']} [source: {m['source']}]"
        for m in matches
    )


def _maybe_remember_fact(user_id: str, user_message: str) -> None:
    """
    Very simple heuristic-based memory writer: if the user explicitly says
    'remember that ...' or 'I prefer ...', store it as a profile fact.
    Swap this for an LLM-based extractor for smarter, implicit memory capture.
    """
    patterns = [
        r"remember that (.+)",
        r"remember i (.+)",
        r"i prefer (.+)",
        r"my favorite (\w+) is (.+)",
    ]
    lowered = user_message.lower().strip()
    for pat in patterns:
        m = re.search(pat, lowered)
        if m:
            fact_value = m.group(0)
            key = f"note_{len(fact_value) % 1000}"
            memory.upsert_profile_fact(user_id, key, fact_value)
            return


def _to_gemini_contents(history: list, user_message: str) -> list:
    """Gemini uses role 'model' instead of 'assistant', and a parts-list format."""
    contents = []
    for turn in history:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=turn["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))
    return contents


def generate_reply(
    user_id: str,
    user_message: str,
    rag_chunks: list = None,
    graph_matches: list = None,
    tool_result: str = None,
) -> str:
    _maybe_remember_fact(user_id, user_message)

    profile_text = memory.format_profile_for_prompt(user_id)
    rag_text = _format_rag_context(rag_chunks or [])
    graph_text = _format_graph_context(graph_matches or [])
    tool_text = tool_result or "No tool was called for this query."

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        profile=profile_text,
        rag_context=rag_text,
        graph_context=graph_text,
        tool_context=tool_text,
    )

    history = memory.get_recent_history(user_id)
    contents = _to_gemini_contents(history, user_message)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=2048,
            # gemini-2.5-flash "thinks" before answering by default, which can eat
            # the whole token budget and leave an empty reply for simple chats.
            # Disabling it keeps this straightforward chatbot fast and reliable.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    reply_text = response.text or ""
    if not reply_text:
        # Surface *why* it was empty instead of silently returning nothing
        try:
            finish_reason = response.candidates[0].finish_reason
        except (IndexError, AttributeError):
            finish_reason = "unknown"
        reply_text = f"(No text returned by the model — finish_reason: {finish_reason})"

    memory.save_message(user_id, "user", user_message)
    memory.save_message(user_id, "assistant", reply_text)

    return reply_text
