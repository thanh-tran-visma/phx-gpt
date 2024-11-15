import logging
from typing import Union, List

from llama_cpp import (
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestUserMessage,
    Llama,
)
from starlette.concurrency import run_in_threadpool


def validate_message(message):
    # Manually check for required keys or structure
    if not isinstance(message, dict):
        return False

    # Check if the message has required fields
    required_keys = [
        "role",
        "content",
    ]  # Replace with the actual required keys for your TypedDicts
    return all(key in message for key in required_keys)


async def get_blue_vi_response(
    llm: Llama,
    messages: Union[
        List[ChatCompletionRequestAssistantMessage],
        List[ChatCompletionRequestUserMessage],
    ],
) -> dict:
    try:
        # Validate the messages manually
        if not all(validate_message(msg) for msg in messages):
            raise ValueError(
                "Messages must be valid dictionaries with required keys."
            )

        response = await run_in_threadpool(
            lambda: llm.create_chat_completion(messages=messages)
        )
        return response
    except Exception as e:
        logging.error(f"Error communicating with the model: {e}")
        return {}
