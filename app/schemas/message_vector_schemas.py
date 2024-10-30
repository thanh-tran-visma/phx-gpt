from pydantic import BaseModel, ConfigDict


class MessageVectorBase(BaseModel):
    id: int
    message_id: int
    tfidf_vector: bytes  # Use bytes for BLOB data
    embedding_vector: bytes  # Use bytes for BLOB data

    model_config = ConfigDict(from_attributes=True)
