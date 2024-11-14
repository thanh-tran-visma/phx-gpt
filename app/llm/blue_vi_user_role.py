import logging
from typing import List
from starlette.concurrency import run_in_threadpool

from llama_cpp import Llama, ChatCompletionRequestUserMessage
from app.model import Message
from app.schemas import GptResponseSchema
from app.types.enum import HTTPStatus


class BlueViGptUserRole:
    def __init__(self, llm: Llama):
        self.llm = llm

    async def get_chat_response(
        self, conversation_history: List[Message]
    ) -> GptResponseSchema:
        """Generate a response from the model based on conversation history for the user role."""
        try:
            # Prepare messages for the model
            mapped_messages: List[ChatCompletionRequestUserMessage] = [
                ChatCompletionRequestUserMessage(
                    role=msg.role, content=msg.content
                )
                for msg in conversation_history
            ]
            response = await run_in_threadpool(
                lambda: self.llm.create_chat_completion(
                    messages=mapped_messages
                )
            )
            choices = response.get("choices")

            if isinstance(choices, list) and len(choices) > 0:
                message_content = choices[0]["message"]["content"]
                return GptResponseSchema(
                    status=HTTPStatus.OK.value, content=message_content
                )

            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content="Sorry, I couldn't generate a response.",
            )
        except Exception as e:
            logging.error(
                f"Unexpected error while generating chat response: {e}"
            )
            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content="Sorry, something went wrong while generating a response. Please retry with a shorter prompt.",
            )
