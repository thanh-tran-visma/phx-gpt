from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class Message:
    role: Literal["user"]
    content: str


@dataclass
class Response:
    content: str
    tokens_used: Optional[int] = None
    prompt_length: Optional[int] = None
