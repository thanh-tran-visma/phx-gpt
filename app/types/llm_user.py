from dataclasses import dataclass
from typing import Literal


@dataclass
class UserPrompt:
    role: Literal["user"]
    content: str
