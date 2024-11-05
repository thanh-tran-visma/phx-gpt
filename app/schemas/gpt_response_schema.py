from typing import Optional
from pydantic import BaseModel


class GptResponseSchema(BaseModel):
    content: str
    tokens_used: Optional[int] = None
    prompt_length: Optional[int] = None
