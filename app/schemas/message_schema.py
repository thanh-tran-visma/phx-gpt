from pydantic import BaseModel
from typing import List
from app.types.enum import MessageType, Role


class MessageSchema(BaseModel):
    conversation_id: int
    content: str
    message_type: MessageType
    role: Role
    embedding_vector: List[float]

    class Config:
        orm_mode = True
