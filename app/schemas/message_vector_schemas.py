from typing import List

from pydantic import BaseModel, ConfigDict


class MessageVectorBase(BaseModel):
    id: int
    message_id: int
    embedding_vector: List[float]

    model_config = ConfigDict(from_attributes=True)
