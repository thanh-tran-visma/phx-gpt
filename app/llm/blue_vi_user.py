import logging
from typing import List
from llama_cpp import Llama
from app.model import Message
from app.schemas import GptResponseSchema
from app.types.enum.http_status import HTTPStatus
from app.types.enum.instruction.blue_vi_gpt_instruction_enum import (
    BlueViInstructionEnum,
)
from app.utils import (
    map_conversation_to_messages,
    get_blue_vi_response,
    convert_blue_vi_response_to_schema,
)
from app.utils.generate_instruction_message import generate_instruction_message


class BlueViGptUserManager:
    def __init__(self, llm: Llama):
        self.llm = llm

    async def get_chat_response(
        self, conversation_history: List[Message]
    ) -> GptResponseSchema:
        """Generate a response from the model based on conversation history for the user role."""
        try:
            mapped_messages = map_conversation_to_messages(
                conversation_history
            )
            response = await get_blue_vi_response(
                self.llm,
                [
                    generate_instruction_message(
                        BlueViInstructionEnum.SYSTEM_DEFAULT_INSTRUCTION.value
                    )
                ]
                + mapped_messages,
            )
            return convert_blue_vi_response_to_schema(response)

        except Exception as e:
            logging.error(
                f"Unexpected error while generating chat response: {e}"
            )
            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content="Sorry, something went wrong while generating a response. Please retry with a shorter prompt.",
            )
