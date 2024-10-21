from typing import Optional
from pydantic import BaseModel, ConfigDict
from datetime import datetime

class ConversationBase(BaseModel):
    id: int
    user_id: int
    created_at: datetime
    end_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)