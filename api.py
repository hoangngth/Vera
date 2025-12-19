from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from authorization import verify_api_key
from pydantic import BaseModel
from vera_core import VeraEngine
from speech_to_text_whisper import transcribe_audio
import uuid
from typing import Optional

app = FastAPI(title="Vera API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins
    allow_credentials=False,      # MUST be False when allow_origins=["*"]
    allow_methods=["*"],          # Allow all HTTP methods
    allow_headers=["*"],          # Allow all headers
)

# In-memory session store
sessions: dict[str, VeraEngine] = {}

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str

class AudioRequest:
    def __init__(
        self,
        file: UploadFile = File(...),
        session_id: str | None = Form(None)
    ):
        self.file = file
        self.session_id = session_id

class AudioResponse(BaseModel):
    session_id: str
    transcript: str
    response: str
    response_audio_url: Optional[str] = None # TTS URL placeholder

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
def chat(req: ChatRequest) -> ChatResponse:

    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = VeraEngine()

    engine = sessions[session_id]
    response = engine.generate_response(req.message)

    return {
        "session_id": session_id,
        "response": response
    }

@app.post("/audio", response_model=AudioResponse, dependencies=[Depends(verify_api_key)])
async def audio_chat(req: AudioRequest = Depends()) -> AudioResponse:

    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = VeraEngine()

    if not req.file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=415,
            detail="File must be an audio file"
        )
    
    audio_bytes = await req.file.read()
    transcript = transcribe_audio(audio_bytes)

    engine = sessions[session_id]
    response = engine.generate_response(transcript)

    return AudioResponse(
        session_id=session_id,
        transcript=transcript,
        response=response,
        audio_url=None
    )