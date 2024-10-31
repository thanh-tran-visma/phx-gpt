from dataclasses import dataclass

from app.types.enum import Role


@dataclass
class UserPrompt:
    role: Role.USER
    content: str
