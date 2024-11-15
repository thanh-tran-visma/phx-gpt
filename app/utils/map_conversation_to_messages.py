from typing import List, Literal, Union
from llama_cpp import (
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestUserMessage,
)
from app.model import Message
from app.types.enum.gpt import Role


def map_conversation_to_messages(
    conversation_history: List[Message],
    role_type: Literal["user", "assistant"],
) -> Union[
    List[ChatCompletionRequestAssistantMessage],
    List[ChatCompletionRequestUserMessage],
]:
    """
    Maps a list of conversation messages to either ChatCompletionRequestAssistantMessage
    or ChatCompletionRequestUserMessage format.

    Args:
        conversation_history (List[Message]): History of conversation messages.
        role_type (Literal["user", "assistant"]): The role type to map ("user" or "assistant").

    Returns:
        Union[
            List[ChatCompletionRequestAssistantMessage],
            List[ChatCompletionRequestUserMessage]
        ]: Mapped messages in the required format based on role_type.
    """
    return [
        (
            ChatCompletionRequestAssistantMessage(
                role=Role.ASSISTANT.value, content=msg.content
            )
            if role_type == Role.ASSISTANT.value
            else ChatCompletionRequestUserMessage(
                role=Role.USER.value, content=msg.content
            )
        )
        for msg in conversation_history
    ]
