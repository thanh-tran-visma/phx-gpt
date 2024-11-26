from typing import Optional, Dict, Any

from pydantic import BaseModel


class GptResponseSchema(BaseModel):
    status: int
    content: str = "No response generated."
    dynamic_json: Optional[Dict[str, Any]] = None
    time_taken: Optional[float] = 0.0
    type: Optional[str] = ''

    class Config:
        extra = 'allow'
