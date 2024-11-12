import json
import logging
from typing import List

from llama_cpp import (
    Llama,
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestUserMessage,
)

from app.model import Message
from app.schemas import GptResponseSchema, PhxAppOperation
from app.types.enum import InstructionTypes


class BlueViGptAssistantRole:
    def __init__(self, llm: Llama):
        self.llm = llm

    def check_for_personal_data(self, prompt: str) -> bool:
        """Detect personal data in the prompt."""
        instruction = (
            f"Detect personal data:\n{prompt}\n"
            f"Return True or False. Note that the data may contain inaccuracies in the response"
        )
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    ChatCompletionRequestAssistantMessage(
                        role="assistant", content=instruction
                    )
                ]
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

    def get_anonymized_message(self, user_message: str) -> GptResponseSchema:
        """Anonymize the user message."""
        instruction = f"Anonymize the data:\n{user_message}\n"

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    ChatCompletionRequestAssistantMessage(
                        role="assistant", content=instruction
                    )
                ]
            )
            choices = response.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                message_content = choices[0]["message"]["content"]
                return GptResponseSchema(content=message_content)
            else:
                return GptResponseSchema(
                    content="Sorry, I couldn't generate an anonymized response."
                )
        except Exception as e:
            logging.error(f"Error generating anonymized message: {e}")
            return GptResponseSchema(
                content="Sorry, an error occurred while generating an anonymized response."
            )

    def identify_instruction_type(self, prompt: str) -> str:
        """Identify the type of instruction based on the prompt content."""
        instruction_prompt = f"Choose the most appropriate instruction between {InstructionTypes.OPERATION.value} and {InstructionTypes.DEFAULT.value} based on the context provided in: {prompt}"
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    ChatCompletionRequestAssistantMessage(
                        role="assistant", content=instruction_prompt
                    )
                ]
            )
            choices = response.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                return choices[0]["message"]["content"].strip()
            return InstructionTypes.DEFAULT.value
        except Exception as e:
            logging.error(f"Error identifying instruction type: {e}")
            return InstructionTypes.DEFAULT.value

    def handle_operation_instructions(
        self, conversation_history: List[Message]
    ) -> PhxAppOperation:  # Return type is PhxAppOperation
        """Generate a response from the model for operation instructions, filling in empty values."""
        try:
            # Prepare the messages for the model, ensuring the conversation history is properly mapped
            mapped_messages: List[ChatCompletionRequestUserMessage] = [
                ChatCompletionRequestUserMessage(
                    role=msg.role, content=msg.content
                )
                for msg in conversation_history
            ]

            # Define the operation schema with empty values for the model to fill
            operation_schema = PhxAppOperation(
                name="",
                invoice_description=None,
                duration=None,
                wizard=None,
                invoicing=False,
                hourlyRate=None,
                unitPrice=None,
                operationRateType=None,
                methodsOfConsult=None,
            )

            # Use model_dump instead of dict to serialize the schema
            operation_schema_dict = operation_schema.model_dump(
                exclude_unset=True
            )
            operating_instructions = {
                "role": "assistant",
                "content": f"Instructions: Use the schema with empty values: {json.dumps(operation_schema_dict)}. Return the correct JSON format. Fields can be None if not in the prompt.",
            }
            mapped_messages.insert(0, operating_instructions)
            response = self.llm.create_chat_completion(
                messages=mapped_messages
            )
            choices = response.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                message_content = choices[0]["message"]["content"]
                try:
                    parsed_response = json.loads(message_content)
                    operation_schema.name = parsed_response.get("name", "")
                    operation_schema.invoice_description = parsed_response.get(
                        "invoice_description", None
                    )
                    operation_schema.duration = parsed_response.get(
                        "duration", None
                    )
                    operation_schema.wizard = parsed_response.get(
                        "wizard", None
                    )
                    operation_schema.invoicing = parsed_response.get(
                        "invoicing", False
                    )
                    operation_schema.hourlyRate = parsed_response.get(
                        "hourlyRate", None
                    )
                    operation_schema.unitPrice = parsed_response.get(
                        "unitPrice", None
                    )
                    operation_schema.operationRateType = parsed_response.get(
                        "operationRateType", None
                    )
                    operation_schema.methodsOfConsult = parsed_response.get(
                        "methodsOfConsult", None
                    )
                except json.JSONDecodeError:
                    logging.error(
                        "Failed to parse the response content as JSON"
                    )
                return operation_schema
            return operation_schema
        except Exception as e:
            logging.error(
                f"Unexpected error while generating chat response: {e}"
            )
            return PhxAppOperation(
                name="Error",
                invoice_description=None,
                duration=None,
                wizard=None,
                invoicing=False,
                hourlyRate=None,
                unitPrice=None,
                operationRateType=None,
                methodsOfConsult=None,
            )
