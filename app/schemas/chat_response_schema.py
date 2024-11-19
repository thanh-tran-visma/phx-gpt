from typing import Optional, Dict, Any
from pydantic import BaseModel


class ChatResponseSchema(BaseModel):
    status: int
    response: str
    conversation_order: Optional[int] = None
    dynamic_json: Optional[Dict[str, Any]] = None

    class Config:
        extra = 'allow'
