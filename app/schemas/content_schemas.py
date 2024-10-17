from pydantic import BaseModel
from datetime import datetime

class ContentBase(BaseModel):
    id: int
    content: str
    created_at: datetime

    class Config:
        from_attributes = True
