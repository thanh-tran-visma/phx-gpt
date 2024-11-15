import logging
from typing import List, Union

from llama_cpp import (
    Llama,
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestUserMessage,
)
from starlette.concurrency import run_in_threadpool


async def get_blue_vi_response(
    llm: Llama,
    messages: Union[
        List[ChatCompletionRequestAssistantMessage],
        List[ChatCompletionRequestUserMessage],
    ],
) -> dict:
    """
    Utility function to request a response from the model.

    Args:
        llm (Llama): The Llama instance used for generating completions.
        messages (Union[List[ChatCompletionRequestAssistantMessage], List[ChatCompletionRequestUserMessage]]):
            A list of assistant or user messages to send to the model.

    Returns:
        dict: The response from the model.
    """
    try:
        # Ensure messages are of the correct type
        if not all(
            isinstance(
                msg,
                (
                    ChatCompletionRequestAssistantMessage,
                    ChatCompletionRequestUserMessage,
                ),
            )
            for msg in messages
        ):
            raise ValueError(
                "Messages must be instances of ChatCompletionRequestAssistantMessage or ChatCompletionRequestUserMessage"
            )

        response = await run_in_threadpool(
            lambda: llm.create_chat_completion(messages=messages)
        )
        return response
    except Exception as e:
        logging.error(f"Error communicating with the model: {e}")
        return {}
