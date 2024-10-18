from pydantic import BaseModel, ConfigDict
from datetime import datetime

class ContentBase(BaseModel):
    id: int
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
