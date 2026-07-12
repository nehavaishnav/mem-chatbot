from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import orchestrator

app = FastAPI(title="Memory-Augmented Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real domain once deployed, if needed
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    route: str
    sources_used: list


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        result = orchestrator.run_chat(req.user_id, req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sources = sorted(set(c["source"] for c in result.get("rag_chunks") or []))
    sources += sorted(set(m["source"] for m in result.get("graph_matches") or []))

    return ChatResponse(
        reply=result["reply"],
        route=result.get("route", "unknown"),
        sources_used=sorted(set(sources)),
    )


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
