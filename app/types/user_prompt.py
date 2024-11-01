from dataclasses import dataclass
from typing import Optional

@dataclass
class UserPrompt:
    prompt: str
    user_id: int
    conversation_id: Optional[int] = None
