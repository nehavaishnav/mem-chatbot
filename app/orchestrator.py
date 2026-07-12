"""
Dynamic Intelligence Layer (LangGraph)
-----------------------------------------
This is the router described in the project brief: for every incoming
message, decide whether to answer using RAG (static knowledge), the
Knowledge Graph (structured relationships), a live tool (real-time
data), or just chat directly — then generate the final answer.

Graph shape:

    router --route=rag--> rag_node   ---\
           --route=graph-> graph_node ---+--> answer_node --> END
           --route=tool--> tool_node  ---/
           --route=chat--> chat_node  --/
"""

import json
from typing import Optional, TypedDict

from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app import rag, tools, llm
from app import graph as kg

from langgraph.graph import StateGraph, END

_client = genai.Client(api_key=GEMINI_API_KEY)

_rag_index_cache = None


class ChatState(TypedDict):
    user_id: str
    message: str
    route: str
    tool_name: Optional[str]
    tool_args: Optional[dict]
    rag_chunks: list
    graph_matches: list
    tool_result: Optional[str]
    reply: Optional[str]


ROUTER_PROMPT = """Decide how to answer the user's message below. Respond with ONLY a JSON object, no other text, no markdown fences.

Routes:
- "rag": question is answerable from a text knowledge base (policies, FAQs, "what/how much/when" questions about documented facts)
- "graph": question asks about a RELATIONSHIP between two named things (e.g. "what does X include", "what is connected to Y")
- "tool": question needs LIVE/real-time data. Available tools:
{tools_desc}
- "chat": general conversation, greetings, or anything not covered above

Respond exactly as: {{"route": "rag" | "graph" | "tool" | "chat", "tool_name": "<tool name or null>", "tool_args": {{}}}}

User message: {message}
"""


def _get_rag_index():
    global _rag_index_cache
    if _rag_index_cache is None:
        _rag_index_cache = rag.load_index()
    return _rag_index_cache


def _route_node(state: ChatState) -> ChatState:
    prompt = ROUTER_PROMPT.format(tools_desc=tools.tools_description(), message=state["message"])
    response = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=256,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = (response.text or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = {}

    state["route"] = parsed.get("route") or "rag"
    state["tool_name"] = parsed.get("tool_name")
    state["tool_args"] = parsed.get("tool_args") or {}
    return state


def _rag_node(state: ChatState) -> ChatState:
    try:
        index = _get_rag_index()
        state["rag_chunks"] = rag.retrieve(state["message"], index)
    except FileNotFoundError:
        state["rag_chunks"] = []
    return state


def _graph_node(state: ChatState) -> ChatState:
    try:
        state["graph_matches"] = kg.query_graph(state["message"])
    except Exception:
        # Graph DB may not be configured yet — fail soft rather than breaking chat
        state["graph_matches"] = []
    return state


def _tool_node(state: ChatState) -> ChatState:
    name = state.get("tool_name")
    args = state.get("tool_args") or {}
    state["tool_result"] = tools.run_tool(name, args) if name else "No tool was specified."
    return state


def _chat_node(state: ChatState) -> ChatState:
    return state


def _answer_node(state: ChatState) -> ChatState:
    reply = llm.generate_reply(
        user_id=state["user_id"],
        user_message=state["message"],
        rag_chunks=state.get("rag_chunks") or [],
        graph_matches=state.get("graph_matches") or [],
        tool_result=state.get("tool_result"),
    )
    state["reply"] = reply
    return state


def _select_route(state: ChatState) -> str:
    return state["route"]


_builder = StateGraph(ChatState)
_builder.add_node("router", _route_node)
_builder.add_node("rag", _rag_node)
_builder.add_node("graph", _graph_node)
_builder.add_node("tool", _tool_node)
_builder.add_node("chat", _chat_node)
_builder.add_node("answer", _answer_node)

_builder.set_entry_point("router")
_builder.add_conditional_edges(
    "router",
    _select_route,
    {"rag": "rag", "graph": "graph", "tool": "tool", "chat": "chat"},
)
_builder.add_edge("rag", "answer")
_builder.add_edge("graph", "answer")
_builder.add_edge("tool", "answer")
_builder.add_edge("chat", "answer")
_builder.add_edge("answer", END)

compiled_graph = _builder.compile()


def run_chat(user_id: str, message: str) -> dict:
    """Runs the full LangGraph flow for one message. Returns the final state dict."""
    initial_state: ChatState = {
        "user_id": user_id,
        "message": message,
        "route": "",
        "tool_name": None,
        "tool_args": None,
        "rag_chunks": [],
        "graph_matches": [],
        "tool_result": None,
        "reply": None,
    }
    return compiled_graph.invoke(initial_state)
