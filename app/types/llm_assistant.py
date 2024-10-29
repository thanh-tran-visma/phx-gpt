from dataclasses import dataclass
from typing import Optional


@dataclass
class Response:
    content: str
    tokens_used: Optional[int] = None
    prompt_length: Optional[int] = None
