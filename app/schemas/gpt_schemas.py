from pydantic import BaseModel
from datetime import datetime

class GptBase(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
