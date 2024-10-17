from pydantic import BaseModel
from datetime import datetime

class ConversationBase(BaseModel):
    id: int
    thread_id: str
    sender_id: int
    gpt_sender_id: int
    content_id: int
    created_at: datetime
    end_at: datetime

    class Config:
        from_attributes = True
