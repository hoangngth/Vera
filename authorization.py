from fastapi import Header, HTTPException
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("VERA_API_KEY")

def verify_api_key(authorization: str = Header(...)):
    if not API_KEY:
        raise RuntimeError("VERA_API_KEY not set")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")

    token = authorization.split(" ", 1)[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")