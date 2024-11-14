from pydantic import BaseModel
from typing import Optional


class UserPromptSchema(BaseModel):
    uuid: str
    prompt: Optional[str] = None
    conversation_order: Optional[int] = None
