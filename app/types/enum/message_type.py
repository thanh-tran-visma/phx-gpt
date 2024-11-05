from enum import Enum


class MessageType(str, Enum):
    PROMPT = "prompt"
    RESPONSE = "response"
