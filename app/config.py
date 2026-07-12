import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "memory_chatbot")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

RAG_DOCS_DIR = os.getenv("RAG_DOCS_DIR", "data/sample_docs")
RAG_INDEX_PATH = os.getenv("RAG_INDEX_PATH", "storage/rag_index.pkl")

# How many past turns to pull into the prompt as short-term context
CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "6"))
# How many retrieved chunks to include per query
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "3"))
