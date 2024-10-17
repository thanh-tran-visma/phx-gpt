from pydantic import BaseModel, ConfigDict
from datetime import datetime

class ContentBase(BaseModel):
    id: int
    content: str
    created_at: datetime

    class Config(ConfigDict):
        from_attributes = True
