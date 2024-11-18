from llama_cpp import ChatCompletionRequestUserMessage

from app.types.enum.gpt import Role


def generate_instruction_message(
    instruction_content: str,
) -> ChatCompletionRequestUserMessage:
    """
    Utility function to generate instruction message from a string.
    """
    return ChatCompletionRequestUserMessage(
        role=Role.USER.value, content=instruction_content
    )
