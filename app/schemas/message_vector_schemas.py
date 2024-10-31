from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, Integer, JSON


class MessageVector(BaseModel):
    __tablename__ = 'message_vectors'

    message_id = Column(Integer, primary_key=True)
    embedding_vector = Column(JSON)

    model_config = ConfigDict(from_attributes=True)
