"""
app.py
------
FastAPI deployment entry point.

Run with:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Test with:
    POST http://localhost:8000/chat

"""

from uuid import uuid4
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline import pipeline
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates




app = FastAPI(
    title       = "Mental Health Support Chatbot",
    description = "RAG-based mental health support system with language detection, emotion classification, and intent routing.",
    version     = "1.0.0",
)

# ── In-memory session store ──────────────────────────────────────────────────
# Stores conversation history per session_id
_sessions: dict[str, list] = {}


# ── Request / Response schemas ───────────────────────────────────────────────
class ChatRequest(BaseModel):
    message    : str
    session_id : str | None = None   # None = start a new session




class ChatResponse(BaseModel):
    response   : str
    intent     : str
    emotion    : str
    language   : str
    session_id : str


# ── Endpoints ────────────────────────────────────────────────────────────────
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="chat.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint.

    - If session_id is provided, the conversation history is retrieved and used.
    - If session_id is None, a new session is created and returned in the response.
    - Each response includes the session_id so the client can continue the conversation.
    """
    try:
        # Get or create session
        session_id = request.session_id or str(uuid4())
        history    = _sessions.get(session_id, [])

        # Run full pipeline
        result = pipeline(
            user_message = request.message,
            history      = history,
        )

        # Save updated history back to session store
        _sessions[session_id] = result["history"]

        return ChatResponse(
            response   = result["response"],
            intent     = result["intent"],
            emotion    = result["emotion"],
            language   = result["language"],
            session_id = session_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clears the conversation history for a given session."""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}
