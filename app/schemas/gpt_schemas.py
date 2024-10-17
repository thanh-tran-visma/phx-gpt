from pydantic import BaseModel, ConfigDict
from datetime import datetime

class GptBase(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config(ConfigDict):
        from_attributes = True
