import logging
from typing import List, Optional
from llama_cpp import Llama
from app.model import Message
from app.schemas import GptResponseSchema
from app.types.enum.gpt import Role
from app.types.enum.http_status import HTTPStatus
from app.types.enum.instruction.blue_vi_gpt_instruction_enum import (
    BlueViInstructionEnum,
)
from app.utils import (
    get_blue_vi_response,
    convert_blue_vi_response_to_schema,
    convert_conversation_history_to_tuples,
)


class BlueViGptUserManager:
    def __init__(self, llm: Llama):
        self.llm = llm

    async def generate_user_response_with_custom_instruction(
        self,
        conversation_history: List[Message],
        instruction: Optional[str] = None,
    ) -> GptResponseSchema:
        """Generate a response from the model based on conversation history for the user role, optionally with a custom instruction."""
        try:
            # Convert the conversation history into the right format (role, content)
            conversation_history_tuples = (
                convert_conversation_history_to_tuples(conversation_history)
            )

            # Use the provided instruction or fall back to the default system instruction
            system_instruction = (
                instruction
                if instruction
                else BlueViInstructionEnum.BLUE_VI_SYSTEM_DEFAULT_INSTRUCTION.value
            )

            # Generate the response
            response = await get_blue_vi_response(
                self.llm,
                [(Role.SYSTEM.value, system_instruction)]
                + conversation_history_tuples,
            )

            # Convert and return the response as GptResponseSchema
            return convert_blue_vi_response_to_schema(response)

        except Exception as e:
            logging.error(
                f"Unexpected error while generating chat response: {e}"
            )
            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content="Sorry, something went wrong while generating a response.",
            )
