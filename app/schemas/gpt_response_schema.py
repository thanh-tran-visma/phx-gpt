from typing import Optional, Dict, Any

from pydantic import BaseModel


class GptResponseSchema(BaseModel):
    status: int
    content: str = "No response generated."
    dynamic_json: Optional[Dict[str, Any]] = None
    type: Optional[str] = ''

    class Config:
        extra = 'allow'
