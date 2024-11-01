from dataclasses import dataclass

@dataclass
class ChatResponse:
    status: int
    response: str
