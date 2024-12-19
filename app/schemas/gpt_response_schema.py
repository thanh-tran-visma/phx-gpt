from typing import Optional, Dict, Any

from pydantic import BaseModel

from app.types.enum.instruction import InstructionList


class GptResponseSchema(BaseModel):
    status: int
    content: str = "No response generated."
    dynamic_json: Optional[Dict[str, Any]] = None
    time_taken: Optional[float] = 0.0
    operationType: Optional[str] = InstructionList.DEFAULT.value

    class Config:
        extra = 'allow'
