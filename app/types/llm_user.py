from dataclasses import dataclass
from typing import Literal


@dataclass
class Prompt:
    role: Literal["user"]
    content: str
