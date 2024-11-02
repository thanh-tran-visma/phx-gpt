from pydantic import BaseModel
from typing import Optional


class UserPromptSchema(BaseModel):
    prompt: str
    user_id: int
    conversation_id: Optional[int] = None
