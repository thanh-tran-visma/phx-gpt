import json
import logging
from typing import List

from llama_cpp import Llama
from app.model import Message
from app.schemas import (
    GptResponseSchema,
    PhxAppOperation,
)
from app.types.enum.instruction import TrainingInstructionEnum
from app.types.enum.instruction.blue_vi_gpt_instruction_enum import (
    BlueViInstructionEnum,
)
from app.utils import (
    get_blue_vi_response,
    map_conversation_to_messages,
    convert_blue_vi_response_to_schema,
)
from app.utils.generate_instruction_message import generate_instruction_message


class BlueViGptAssistant:
    def __init__(self, llm: Llama):
        self.llm = llm

    async def check_for_personal_data(self, prompt: str) -> bool:
        """Detect personal data in the prompt relevant to GDPR compliance."""
        response = await get_blue_vi_response(
            self.llm,
            [
                generate_instruction_message(
                    BlueViInstructionEnum.BLUE_VI_FLAG_GDPR_INSTRUCTION.value
                )
            ]
            + [generate_instruction_message(prompt)],
        )
        if not response:
            return False
        result = convert_blue_vi_response_to_schema(response)
        return "True" in result.content

    async def get_anonymized_message(
        self, user_message: str
    ) -> GptResponseSchema:
        """Anonymize the user message."""
        instruction = (
            f"{BlueViInstructionEnum.BLUE_VI_ASSISTANT_ANONYMIZE_DATA.value:}"
        )
        response = await get_blue_vi_response(
            self.llm,
            [generate_instruction_message(instruction)]
            + [generate_instruction_message(user_message)],
        )

        return convert_blue_vi_response_to_schema(response)

    async def identify_instruction_type(self, prompt: str) -> str:
        """
        Identify the type of instruction based on the prompt content.
        """
        instruction = (
            f"Choose the most appropriate instruction between "
            f"{TrainingInstructionEnum.OPERATION_INSTRUCTION.value} and {TrainingInstructionEnum.DEFAULT.value} "
            f"based on the context provided in:"
        )
        response = await get_blue_vi_response(
            self.llm,
            [generate_instruction_message(instruction)]
            + [generate_instruction_message(prompt)],
        )

        if not response:
            return TrainingInstructionEnum.DEFAULT.value

        result = convert_blue_vi_response_to_schema(response)
        return (
            TrainingInstructionEnum.OPERATION_INSTRUCTION.value
            if TrainingInstructionEnum.OPERATION_INSTRUCTION.value
            in result.content
            else TrainingInstructionEnum.DEFAULT.value
        )

    async def get_operation_format(
        self, conversation_history: List[Message]
    ) -> PhxAppOperation:
        """Generates an operation schema based on the user's conversation history and model response."""
        # Prepare conversation messages for the model
        model_messages = map_conversation_to_messages(conversation_history)
        instruction = TrainingInstructionEnum.ASSISTANT_OPERATION_HANDLING.value
        # Get response from LLM
        response = await get_blue_vi_response(
            self.llm,
            [generate_instruction_message(instruction)] + model_messages,
        )
        result = convert_blue_vi_response_to_schema(response)
        logging.info(response)
        logging.info('response')
        logging.info('before json loads')
        logging.info(result.content)
        try:
            data: PhxAppOperation = json.loads(result.content)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
            return PhxAppOperation()
        if data:return data
        else: return PhxAppOperation()
