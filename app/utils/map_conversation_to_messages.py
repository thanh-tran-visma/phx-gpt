from typing import List
from llama_cpp import (
    ChatCompletionRequestUserMessage,
)
from app.model import Message
from app.types.enum.gpt import Role


def map_conversation_to_messages(
    conversation_history: List[Message],
) -> [List[ChatCompletionRequestUserMessage]]:
    """
    Maps a list of conversation messages to ChatCompletionRequestUserMessage format.

    Args:
        conversation_history (List[Message]): History of conversation messages.

    Returns: [List[ChatCompletionRequestUserMessage]]
    """
    return [
        (
            ChatCompletionRequestUserMessage(
                role=Role.USER.value, content=msg.content
            )
        )
        for msg in conversation_history
    ]
