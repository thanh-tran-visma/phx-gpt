from typing import Optional, Dict, Any

from pydantic import BaseModel


class GptResponseSchema(BaseModel):
    status: int
    content: str
    dynamic_json: Optional[Dict[str, Any]] = None

    class Config:
        extra = 'allow'
