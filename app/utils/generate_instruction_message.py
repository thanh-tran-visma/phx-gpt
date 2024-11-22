from llama_cpp import (
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
)
from app.types.enum.gpt import Role


def generate_instruction_message(
    instruction_content: str,
) -> ChatCompletionRequestSystemMessage:
    """
    Utility function to generate system instruction message from a string.
    """
    return ChatCompletionRequestSystemMessage(
        role=Role.SYSTEM.value, content=instruction_content
    )


def generate_user_message(
    instruction_content: str,
) -> ChatCompletionRequestUserMessage:
    """
    Utility function to generate system instruction message from a string.
    """
    return ChatCompletionRequestUserMessage(
        role=Role.USER.value, content=instruction_content
    )
