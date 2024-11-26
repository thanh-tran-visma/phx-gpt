from enum import Enum
from pydantic import BaseModel, ConfigDict
from datetime import datetime


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"


class MessageType(str, Enum):
    prompt = "prompt"
    response = "response"


class MessageBase(BaseModel):
    id: int
    conversation_id: int
    content: str
    message_type: MessageType
    role: MessageRole
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
