# Memory-Augmented Chatbot with Knowledge Graph and Hybrid RAG

A chatbot combining **RAG** (static knowledge), a **Knowledge Graph** (Neo4j,
structured relationships), **long-term memory** (MongoDB), **live tools**
(real-time data), and **LangGraph routing** to decide which of these to use
for each incoming message — matching the full original project brief.

## What's inside

```
memory-chatbot/
├── app/
│   ├── config.py       # loads settings from .env
│   ├── rag.py           # chunking + TF-IDF retrieval (static knowledge)
│   ├── graph.py          # Neo4j knowledge graph: entity/relationship extraction + queries
│   ├── tools.py          # live data tools (weather, datetime)
│   ├── memory.py        # MongoDB: chat history + long-term user facts
│   ├── orchestrator.py   # LangGraph router: decides rag / graph / tool / chat per message
│   ├── llm.py            # builds the final prompt from gathered context and calls Gemini
│   └── main.py           # FastAPI app: POST /chat
├── data/sample_docs/     # sample knowledge base (2 .txt files) — replace with your own
├── ingest.py             # builds the RAG index from data/sample_docs
├── build_graph.py        # builds the Neo4j knowledge graph from data/sample_docs
├── eval.py                # LLM-as-judge evaluation: relevance, correctness, faithfulness
├── chat_cli.py         # terminal chat client, no frontend needed
└── requirements.txt
```

**Why TF-IDF instead of embeddings/FAISS?** For a minimal demo it removes the need
for a GPU or downloading an embedding model, while keeping the exact same pipeline
shape (chunk → vectorize → store → retrieve). `app/rag.py` is written so you can
swap in `sentence-transformers` + FAISS/Chroma later by only changing the
vectorize/search functions — nothing else in the app needs to change.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env`:
   - `GEMINI_API_KEY` — your free Gemini API key ([aistudio.google.com/apikey](https://aistudio.google.com/apikey), no credit card needed)
   - `MONGODB_URI` — your existing MongoDB connection string
   - `MONGODB_DB` — database name (created automatically if it doesn't exist)

3. **Add your knowledge base**
   Drop `.txt` files into `data/sample_docs/` (two sample files are already there —
   company policies and a product FAQ). Any plain text works: FAQs, docs, manuals.

4. **Build the RAG index**
   ```bash
   python ingest.py
   ```
   Re-run this any time you change the files in `data/sample_docs/`.

5. **Test it in the terminal (fastest way to check everything works)**
   ```bash
   python chat_cli.py
   ```
   Try: `"How many vacation days do I get?"` or `"What's the Pro plan price?"`
   Then try: `"Remember that I prefer short answers"` followed by a new question —
   the bot should recall that preference in later replies (check MongoDB's
   `profile` collection to see it stored).

6. **Or run it as an API**
   ```bash
   uvicorn app.main:app --reload
   ```
   Then:
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"user_id": "alice", "message": "How many vacation days do I get?"}'
   ```
   Interactive docs at `http://localhost:8000/docs`.

## How a request flows through the system

1. User sends a message with a `user_id`.
2. `rag.retrieve()` finds the most relevant chunks from your knowledge base.
3. `memory.get_recent_history()` pulls the last few turns of this user's chat.
4. `memory.format_profile_for_prompt()` pulls any long-term facts remembered about them.
5. `llm.generate_reply()` assembles all of this into a system prompt and calls Gemini.
6. The new user message + reply are saved back to MongoDB for next time.

## Deploying it as a live website (free)

The project now includes a simple built-in chat webpage (`static/index.html`), served directly by the FastAPI backend — so there's only one thing to deploy, not a separate frontend and backend.

We'll use **Render** (has a free tier, no credit card needed to start) and **GitHub** (to hold your code so Render can deploy it).

1. **Put your code on GitHub**
   - Create a free account at [github.com](https://github.com) if you don't have one.
   - Create a new repository (e.g. `memory-chatbot`), then upload your project folder to it (GitHub's web UI has an "upload files" option — no command line needed).
   - **Do NOT upload your `.env` file** — it has your secret keys in it. `.gitignore` is already set up to skip it if you use `git push` instead.

2. **Create a Render account and new Web Service**
   - Sign up free at [render.com](https://render.com).
   - Click **New > Web Service**, connect your GitHub account, and select your `memory-chatbot` repo.
   - Render should auto-detect the included `render.yaml` and pre-fill the build/start commands. If not, set them manually:
     - Build command: `pip install -r requirements.txt && python ingest.py`
     - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

3. **Add your environment variables**
   In Render's dashboard for this service, go to **Environment** and add:
   - `GEMINI_API_KEY` — your real key
   - `MONGODB_URI` — your Atlas connection string
   - `MONGODB_DB` — `memory_chatbot`
   - `GEMINI_MODEL` — `gemini-3.5-flash` (or whatever current model you're using)

4. **Deploy**
   Click **Create Web Service**. Render will build and start it — this takes a few minutes on first deploy. When it's done, you'll get a live URL like `https://memory-chatbot-xxxx.onrender.com`. Open it in a browser — that's your chatbot, live on the internet.

**Note on the free tier:** Render's free web services "sleep" after 15 minutes of no traffic and take ~30–60 seconds to wake back up on the next visit. That's normal and fine for a demo/portfolio project.

## The full system (Knowledge Graph + LangGraph + Tools + Evaluation)

This project now matches the full original brief, not just the minimal RAG+memory slice:

```
app/
├── rag.py           # Static Knowledge Layer — text search
├── graph.py         # Knowledge Graph Layer — Neo4j, entity relationships
├── tools.py         # Dynamic Tools — live weather/datetime lookups (no key needed)
├── memory.py        # Long-term memory — MongoDB
├── orchestrator.py  # LangGraph router — decides rag / graph / tool / chat per message
├── llm.py           # Builds the final prompt from whatever context was gathered, calls Gemini
build_graph.py         # Builds the Neo4j knowledge graph from data/sample_docs
eval.py                 # LLM-as-judge evaluation: relevance, correctness, faithfulness
```

### How a message flows now

1. `orchestrator.run_chat()` sends your message to a **router** node, which asks Gemini to classify it as `rag`, `graph`, `tool`, or `chat`.
2. Based on that route, LangGraph calls exactly one of: RAG retrieval, a Knowledge Graph lookup, a live tool (e.g. weather), or nothing (plain chat).
3. Whatever context was gathered — plus your long-term memory — is handed to `llm.generate_reply()` to produce the final answer.
4. The chat response now also returns which `route` was chosen, shown in the web UI.

### Setting up the Knowledge Graph (Neo4j)

You need a running Neo4j instance. The free option is **Neo4j Aura**:

1. Sign up free at [neo4j.com/cloud/aura-free](https://neo4j.com/cloud/aura-free/), create a free instance.
2. Aura gives you a connection URI (`neo4j+s://...`), username (usually `neo4j`), and a generated password — save all three.
3. Put them in `.env`:
   ```
   NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_generated_password
   ```
4. Build the graph (extracts entities/relationships from `data/sample_docs` using Gemini, one-time per doc change):
   ```bash
   python build_graph.py
   ```
5. Try a relationship-style question in `chat_cli.py` or the web UI, e.g. *"What does the Pro plan include?"* — the router should pick `graph` and pull in the relevant relationships.

**On Render:** add `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` as environment variables (already wired into `render.yaml`), and the build command now also runs `build_graph.py` automatically on every deploy.

### Trying the live tool

Ask something like *"What's the weather in Paris right now?"* — the router should pick `tool`, call the free Open-Meteo API, and answer with a real current reading. No API key needed for this one.

### Running the evaluation framework

```bash
python eval.py
```
This runs a small built-in test set through the full pipeline and prints a table scoring each answer on **context relevance**, **answer correctness**, and **faithfulness** (0–1, judged by Gemini), plus whether the router picked the expected route. Edit `TEST_CASES` in `eval.py` to test your own knowledge base once you swap in real documents.

## What's still optional beyond this

Two pieces from the original brief are left as easy extensions rather than built in, since they depend entirely on your real data source:

| Piece | Where it plugs in |
|---|---|
| Real embeddings + FAISS/Chroma (instead of TF-IDF) | Replace `TfidfVectorizer`/`cosine_similarity` in `app/rag.py` — nothing else needs to change |
| Web scraping ingestion | Add a `scrape.py` (BeautifulSoup/Scrapy) that writes `.txt` files into `data/sample_docs/`, then `ingest.py` and `build_graph.py` run as-is on the new content |

## Notes

- The memory-writing logic in `llm.py` (`_maybe_remember_fact`) is a simple regex
  heuristic ("remember that...", "I prefer..."). For smarter, implicit memory
  capture, replace it with a small Gemini call that decides what's worth
  remembering from each turn.
- The Knowledge Graph's relationship lookup (`app/graph.py`'s `query_graph`) uses
  simple keyword matching rather than a full NL-to-Cypher layer — good enough for
  a demo-scale graph, but worth upgrading to an LLM-generated Cypher query if your
  graph grows large.
- `chat_history` and `profile` are separate MongoDB collections so short-term
  conversational context and durable preferences can be tuned independently
  (e.g. different retention windows).
