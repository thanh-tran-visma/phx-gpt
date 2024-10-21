from pydantic import BaseModel, ConfigDict
from datetime import datetime

class MessageBase(BaseModel):
    id: int
    conversation_id: int
    content: str
    message_type: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)