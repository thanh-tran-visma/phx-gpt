from pydantic import BaseModel, ConfigDict
from datetime import datetime

class ConversationBase(BaseModel):
    id: int
    thread_id: str
    sender_id: int
    gpt_sender_id: int
    content_id: int
    created_at: datetime
    end_at: datetime

    model_config = ConfigDict(from_attributes=True)
