import json
import logging
from json import JSONDecodeError
from typing import List
from llama_cpp import Llama, ChatCompletionRequestAssistantMessage
from app.schemas import GptResponseSchema, PhxAppOperation
from app.types.enum.gpt import Role
from app.utils import (
    get_blue_vi_response,
    map_conversation_to_messages,
    convert_blue_vi_response_to_schema,
)
from app.types.enum.instruction import InstructionEnum
from app.types.enum.operation import OperationRateType, VatRate
from app.model import Message


class BlueViGptAssistantRole:
    def __init__(self, llm: Llama):
        self.llm = llm

    async def check_for_personal_data(self, prompt: str) -> bool:
        """Detect personal data in the prompt."""
        instruction = f"Detect personal data:\n{prompt}\nReturn True or False. Note that the data may contain inaccuracies in the response"

        response = await get_blue_vi_response(
            self.llm,
            [
                ChatCompletionRequestAssistantMessage(
                    role=Role.ASSISTANT.value, content=instruction
                )
            ],
        )

        if not response:
            logging.warning("Error detecting personal data.")
            return False

        result = convert_blue_vi_response_to_schema(response)
        return "True" in result.content

    async def get_anonymized_message(
        self, user_message: str
    ) -> GptResponseSchema:
        """Anonymize the user message."""
        instruction = f"{InstructionEnum.Assistant_Anonymize_Data.value}:\n{user_message}\n"

        response = await get_blue_vi_response(
            self.llm,
            [
                ChatCompletionRequestAssistantMessage(
                    role=Role.ASSISTANT.value, content=instruction
                )
            ],
        )

        return convert_blue_vi_response_to_schema(response)

    async def identify_instruction_type(self, prompt: str) -> str:
        """
        Identify the type of instruction based on the prompt content.

        Args:
            prompt (str): The input prompt to analyze.

        Returns:
            str: InstructionEnum.OPERATION_Instruction.value if it is present
                 in the result content; otherwise, InstructionEnum.DEFAULT.value.
        """
        # Construct the instruction to be sent to the model
        instruction = (
            f"Choose the most appropriate instruction between "
            f"{InstructionEnum.OPERATION_Instruction.value} and {InstructionEnum.DEFAULT.value} "
            f"based on the context provided in: {prompt}"
        )

        # Send the instruction to the model and await its response
        response = await get_blue_vi_response(
            self.llm,
            [
                ChatCompletionRequestAssistantMessage(
                    role=Role.ASSISTANT.value, content=instruction
                )
            ],
        )
        if not response:
            return InstructionEnum.DEFAULT.value
        result = convert_blue_vi_response_to_schema(response)
        if InstructionEnum.OPERATION_Instruction.value in result.content:
            return InstructionEnum.OPERATION_Instruction.value
        else:
            return InstructionEnum.DEFAULT.value

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
        model_messages = map_conversation_to_messages(
            conversation_history, role_type=Role.ASSISTANT.value
        )
        instruction_message = {
            "role": Role.ASSISTANT.value,
            "content": f"Instructions: Use the schema with empty values: {operation_schema}. {InstructionEnum.ASSISTANT_OPERATION_HANDLING.value} Fields can be None if not provided in the prompt.",
        }
        model_messages.append(instruction_message)

        # Request response from the model
        response = await get_blue_vi_response(self.llm, model_messages)

        if not response:
            logging.error(
                "Failed to receive valid response for operation format."
            )
            return operation_schema

        # Process the model response and update the schema
        result = convert_blue_vi_response_to_schema(response)
        try:
            parsed_response = json.loads(result.content)
            for field, value in parsed_response.items():
                if hasattr(operation_schema, field):
                    setattr(operation_schema, field, value)
        except (JSONDecodeError, ValueError) as e:
            logging.error(f"Error processing model response: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during operation handling: {e}")

        return operation_schema
