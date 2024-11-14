import json
import logging
from typing import List

from fastapi.encoders import jsonable_encoder
from llama_cpp import (
    Llama,
    ChatCompletionRequestAssistantMessage,
)
from starlette.concurrency import run_in_threadpool

from app.model import Message
from app.schemas import GptResponseSchema, PhxAppOperation
from app.types.enum import HTTPStatus
from app.types.enum.instruction import InstructionEnum
from app.types.enum.operation import OperationRateType, VatRate


class BlueViGptAssistantRole:
    def __init__(self, llm: Llama):
        self.llm = llm

    async def check_for_personal_data(self, prompt: str) -> bool:
        """Detect personal data in the prompt."""
        instruction = (
            f"Detect personal data:\n{prompt}\n"
            f"Return True or False. Note that the data may contain inaccuracies in the response"
        )
        try:
            response = await run_in_threadpool(
                lambda: self.llm.create_chat_completion(
                    messages=[
                        ChatCompletionRequestAssistantMessage(
                            role="assistant", content=instruction
                        )
                    ]
                )
            )
            choices = response.get("choices")

            if isinstance(choices, list) and len(choices) > 0:
                model_response = choices[0]["message"]["content"]
                return "True" in model_response
            else:
                logging.warning("Model did not return a valid response.")
                return False
        except Exception as e:
            logging.error(f"Error checking for personal data: {e}")
            return False

    async def get_anonymized_message(
        self, user_message: str
    ) -> GptResponseSchema:
        """Anonymize the user message."""
        instruction = f"Anonymize the data:\n{user_message}\n"

        try:
            response = await run_in_threadpool(
                lambda: self.llm.create_chat_completion(
                    messages=[
                        ChatCompletionRequestAssistantMessage(
                            role="assistant", content=instruction
                        )
                    ]
                )
            )
            choices = response.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                message_content = choices[0]["message"]["content"]
                return GptResponseSchema(
                    status=HTTPStatus.OK.value, content=message_content
                )
            else:
                return GptResponseSchema(
                    status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                    content="Sorry, I couldn't generate an anonymized response.",
                )
        except Exception as e:
            logging.error(f"Error generating anonymized message: {e}")
            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content="Sorry, an error occurred while generating an anonymized response.",
            )

    async def identify_instruction_type(self, prompt: str) -> str:
        """Identify the type of instruction based on the prompt content."""
        instruction = f"Choose the most appropriate instruction between {InstructionEnum.OPERATION.value} and {InstructionEnum.DEFAULT.value} based on the context provided in: {prompt}"
        try:
            response = await run_in_threadpool(
                lambda: self.llm.create_chat_completion(
                    messages=[
                        ChatCompletionRequestAssistantMessage(
                            role="assistant", content=instruction
                        )
                    ]
                )
            )
            choices = response.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                return choices[0]["message"]["content"].strip()
            return InstructionEnum.DEFAULT.value
        except Exception as e:
            logging.error(f"Error identifying instruction type: {e}")
            return InstructionEnum.DEFAULT.value

    async def handle_operation_instruction(
        self, uuid: str, conversation_history: List[Message]
    ) -> bool:
        """Handle operation instructions"""
        try:
            # Initialize operation schema with default values
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

            logging.info("Handling operation for UUID: %s", uuid)

            # Serialize schema with default values using jsonable_encoder
            operation_schema_dict = jsonable_encoder(
                operation_schema, exclude_unset=True
            )

            logging.debug("Serialized schema: %s", operation_schema_dict)

            # Prepare operating instructions for the LLM
            operating_instructions = {
                "role": "assistant",
                "content": f"Instructions: Use the schema with empty values: {operation_schema_dict}. Return the correct JSON format. Fields can be None if not in the prompt.",
            }

            # Create mapped messages for the model
            mapped_messages: List[ChatCompletionRequestAssistantMessage] = [
                ChatCompletionRequestAssistantMessage(
                    role=msg.role, content=msg.content
                )
                for msg in conversation_history
            ]
            # Include the instruction message as the last message
            mapped_messages.append(operating_instructions)

            # Get response from the model
            response = await run_in_threadpool(
                lambda: self.llm.create_chat_completion(
                    messages=mapped_messages
                )
            )

            logging.debug("Model response: %s", response)

            choices = response.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                message_content = choices[0]["message"]["content"]

                # Check if message_content is a string before parsing
                if isinstance(message_content, str):
                    try:
                        # Parse model's JSON response and update schema fields
                        parsed_response = json.loads(message_content)

                        # Map the parsed response back to operation schema
                        operation_schema.name = parsed_response.get(
                            "name", operation_schema.name
                        )
                        operation_schema.description = parsed_response.get(
                            "description", operation_schema.description
                        )
                        operation_schema.duration = parsed_response.get(
                            "duration", operation_schema.duration
                        )
                        operation_schema.invoicing = parsed_response.get(
                            "invoicing", operation_schema.invoicing
                        )
                        operation_schema.hourlyRate = parsed_response.get(
                            "hourlyRate", operation_schema.hourlyRate
                        )
                        operation_schema.unitPrice = parsed_response.get(
                            "unitPrice", operation_schema.unitPrice
                        )
                        operation_schema.operationRateType = (
                            parsed_response.get(
                                "operationRateType",
                                operation_schema.operationRateType,
                            )
                        )
                        operation_schema.methodsOfConsult = (
                            parsed_response.get(
                                "methodsOfConsult",
                                operation_schema.methodsOfConsult,
                            )
                        )
                        operation_schema.forAppointment = parsed_response.get(
                            "forAppointment", operation_schema.forAppointment
                        )
                        operation_schema.vatRate = parsed_response.get(
                            "vatRate", operation_schema.vatRate
                        )
                    except json.JSONDecodeError as e:
                        logging.error(
                            "Failed to parse the response content as JSON: %s",
                            e,
                        )
                else:
                    logging.error(
                        "The message content is not a valid JSON string: %s",
                        message_content,
                    )

            logging.info(
                "Operation schema after processing: %s", operation_schema
            )
            return True

        except Exception as e:
            logging.error(
                "Unexpected error while handling operation instruction chat response: %s",
                e,
            )
            return False
