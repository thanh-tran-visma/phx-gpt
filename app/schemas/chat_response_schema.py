from typing import Optional

from pydantic import BaseModel


class ChatResponseSchema(BaseModel):
    status: int
    response: str
    conversation_order: Optional[int] = None
