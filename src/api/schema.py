from pydantic import BaseModel
from typing import Optional, Any


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None

class UploadResponse(BaseModel):
    filename: str
    chunks: int

class EvaluateResponse(BaseModel):
    results: list[dict[str, Any]]
    metrics: dict[str, Any]