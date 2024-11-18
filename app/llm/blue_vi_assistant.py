import json
import logging
from json import JSONDecodeError
from typing import List

from llama_cpp import Llama
from app.model import Message
from app.schemas import GptResponseSchema, PhxAppOperation
from app.types.enum.instruction import InstructionEnum
from app.types.enum.instruction.blue_vi_gpt_instruction_enum import (
    BlueViInstructionEnum,
)
from app.types.enum.operation import OperationRateType, VatRate
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
                    BlueViInstructionEnum.FLAG_GDPR_INSTRUCTION.value
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
        instruction = f"{InstructionEnum.Assistant_Anonymize_Data.value:}"
        instruction_messages = [generate_instruction_message(instruction)]
        response = await get_blue_vi_response(
            self.llm,
            instruction_messages
            + [generate_instruction_message(user_message)],
        )

        return convert_blue_vi_response_to_schema(response)

    async def identify_instruction_type(self, prompt: str) -> str:
        """
        Identify the type of instruction based on the prompt content.
        """
        instruction = (
            f"Choose the most appropriate instruction between "
            f"{InstructionEnum.OPERATION_INSTRUCTION.value} and {InstructionEnum.DEFAULT.value} "
            f"based on the context provided in:"
        )
        response = await get_blue_vi_response(
            self.llm,
            [generate_instruction_message(instruction)]
            + [generate_instruction_message(prompt)],
        )

        if not response:
            return InstructionEnum.DEFAULT.value

        result = convert_blue_vi_response_to_schema(response)
        return (
            InstructionEnum.OPERATION_INSTRUCTION.value
            if InstructionEnum.OPERATION_INSTRUCTION.value in result.content
            else InstructionEnum.DEFAULT.value
        )

    async def get_operation_format(
        self, uuid: str, conversation_history: List[Message]
    ) -> PhxAppOperation:
        """Generates an operation schema based on the user's conversation history and model response."""
        operation_schema = PhxAppOperation(
            name="",
            description=None,
            duration=None,
            invoicing=False,
            hourlyRate=None,
            unitPrice=0,
            operationRateType=OperationRateType.UNIT_PRICE,
            methodsOfConsult=[],
            forAppointment=True,
            vatRate=VatRate.LOW,
            uuid=uuid,
        )

        # Prepare conversation messages for the model
        model_messages = map_conversation_to_messages(conversation_history)
        instruction = f"{InstructionEnum.ASSISTANT_OPERATION_HANDLING.value}. Fields can be None if not provided in the prompt. No comments allowed"

        # Get response from LLM
        response = await get_blue_vi_response(
            self.llm,
            [generate_instruction_message(instruction)] + model_messages,
        )

        # Convert response to schema
        result = convert_blue_vi_response_to_schema(response)
        logging.info(f"Raw model response: {result.content}")

        # Process the model response and update the schema
        try:
            parsed_response = json.loads(result.content)
            for field, value in parsed_response.items():
                if hasattr(operation_schema, field):
                    setattr(operation_schema, field, value)
        except JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")

        return operation_schema
