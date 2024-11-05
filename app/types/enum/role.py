from enum import Enum


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
