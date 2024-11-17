import logging
from typing import List

from llama_cpp import (
    ChatCompletionRequestUserMessage,
    Llama,
)
from starlette.concurrency import run_in_threadpool


async def get_blue_vi_response(
    llm: Llama,
    messages: [
        List[ChatCompletionRequestUserMessage],
    ],
) -> dict:
    try:
        response = await run_in_threadpool(
            lambda: llm.create_chat_completion(messages=messages)
        )
        return response
    except Exception as e:
        logging.error(f"Error communicating with the model: {e}")
        return {}
