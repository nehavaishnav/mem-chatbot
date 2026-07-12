"""
Long-Term Memory Layer
-----------------------
Stores two kinds of things per user in MongoDB:

1. `messages` collection — full chat history (every turn), so we can
   pull the last N turns as short-term conversational context.
2. `profile` collection — durable facts/preferences about the user
   ("prefers Python", "works in finance", etc.) that persist across
   sessions, independent of any single conversation.

This is intentionally simple (two collections, plain dicts) so it's easy
to read and extend — e.g. swap the profile extraction for an LLM call
that decides what's worth remembering.
"""

from datetime import datetime, timezone

import certifi
from pymongo import MongoClient

from app.config import MONGODB_URI, MONGODB_DB, CHAT_HISTORY_TURNS

_client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
_db = _client[MONGODB_DB]
messages_col = _db["messages"]
profile_col = _db["profile"]


def save_message(user_id: str, role: str, content: str) -> None:
    messages_col.insert_one(
        {
            "user_id": user_id,
            "role": role,  # "user" or "assistant"
            "content": content,
            "timestamp": datetime.now(timezone.utc),
        }
    )


def get_recent_history(user_id: str, turns: int = CHAT_HISTORY_TURNS) -> list:
    """Returns the last `turns` messages (user+assistant combined), oldest first."""
    cursor = (
        messages_col.find({"user_id": user_id})
        .sort("timestamp", -1)
        .limit(turns)
    )
    history = list(cursor)[::-1]
    return [{"role": m["role"], "content": m["content"]} for m in history]


def upsert_profile_fact(user_id: str, key: str, value: str) -> None:
    """Stores/updates a single durable fact about the user, e.g. key='favorite_language'."""
    profile_col.update_one(
        {"user_id": user_id},
        {"$set": {f"facts.{key}": value, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )


def get_profile(user_id: str) -> dict:
    doc = profile_col.find_one({"user_id": user_id})
    if not doc:
        return {}
    return doc.get("facts", {})


def format_profile_for_prompt(user_id: str) -> str:
    facts = get_profile(user_id)
    if not facts:
        return "No known preferences yet."
    return "\n".join(f"- {k}: {v}" for k, v in facts.items())
