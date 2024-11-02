from pydantic import BaseModel


class ChatResponseSchema(BaseModel):
    status: int
    response: str
