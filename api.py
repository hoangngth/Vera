from fastapi import FastAPI, Depends
from authorization import verify_api_key
from pydantic import BaseModel
from vera_core import VeraEngine
import uuid

app = FastAPI(title="Vera API")

# In-memory session store
sessions: dict[str, VeraEngine] = {}

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = VeraEngine()

    engine = sessions[session_id]
    response = engine.generate_response(req.message)

    return {
        "session_id": session_id,
        "response": response
    }
