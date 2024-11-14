from pydantic import BaseModel


class GptResponseSchema(BaseModel):
    status: int
    content: str
