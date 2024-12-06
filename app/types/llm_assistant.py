from dataclasses import dataclass
from typing import Optional, List


@dataclass
class GptResponse:
    content: str
    tokens_used: Optional[int] = None
    prompt_length: Optional[int] = None
    embedding: Optional[List[float]] = None
