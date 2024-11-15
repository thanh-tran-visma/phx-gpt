import logging
from typing import List
from llama_cpp import Llama
from app.model import Message
from app.schemas import GptResponseSchema
from app.types.enum.http_status import HTTPStatus
from app.utils import (
    map_conversation_to_messages,
    get_blue_vi_response,
    process_model_response,
)
from app.types.enum.gpt import Role


class BlueViGptUserRole:
    def __init__(self, llm: Llama):
        self.llm = llm

    async def get_chat_response(
        self, conversation_history: List[Message]
    ) -> GptResponseSchema:
        """Generate a response from the model based on conversation history for the user role."""
        try:
            mapped_messages = map_conversation_to_messages(
                conversation_history, role_type=Role.USER.value
            )

            # Request the model response using the utility function
            response = await get_blue_vi_response(self.llm, mapped_messages)

            # Process the response using the utility function
            return process_model_response(response)

        except Exception as e:
            logging.error(
                f"Unexpected error while generating chat response: {e}"
            )
            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content="Sorry, something went wrong while generating a response. Please retry with a shorter prompt.",
            )
