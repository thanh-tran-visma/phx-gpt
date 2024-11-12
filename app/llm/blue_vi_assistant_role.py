import json
import logging
from llama_cpp import (
    Llama,
    ChatCompletionRequestAssistantMessage,
)
from app.redis import RedisClient
from app.schemas import GptResponseSchema, PhxAppOperation
from app.types.enum import InstructionTypes


class BlueViGptAssistantRole:
    def __init__(self, llm: Llama):
        self.llm = llm
        self.redis_client = RedisClient()

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

    async def handle_operation_instructions(
        self, uuid: str, user_input: str
    ) -> PhxAppOperation:
        """Generate a response from the model for operation instructions and store the result in Redis."""
        try:
            # Initialize operation schema with new fields
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
                forAppointment=True,
                vatRate=0,
                uuid=uuid,
            )

            # Serialize schema with default values, excluding unset fields
            operation_schema_dict = operation_schema.model_dump(
                exclude_unset=True
            )

            operating_instructions = {
                "role": "assistant",
                "content": f"Instructions: Use the schema with empty values: {json.dumps(operation_schema_dict)}. Return the correct JSON format. Fields can be None if not in the prompt.",
            }

            # Create mapped messages for the model
            mapped_messages = [
                ChatCompletionRequestAssistantMessage(
                    role="assistant", content=user_input
                ),
                operating_instructions,
            ]

            # Get response from the model
            response = self.llm.create_chat_completion(
                messages=mapped_messages
            )
            choices = response.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                message_content = choices[0]["message"]["content"]
                try:
                    # Parse model's JSON response and update schema fields
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
                    operation_schema.forAppointment = parsed_response.get(
                        "forAppointment", True
                    )
                    operation_schema.vatRate = parsed_response.get(
                        "vatRate", 0
                    )

                    # Use UUID directly as Redis key
                    redis_key = f"operation:{uuid}"
                    await self.redis_client.set(
                        redis_key, operation_schema_dict
                    )

                except json.JSONDecodeError:
                    logging.error(
                        "Failed to parse the response content as JSON"
                    )
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
                forAppointment=True,
                vatRate=0,
                uuid=uuid,
            )

    def operation_processing(self, operation: dict) -> GptResponseSchema:
        """Process the operation and check for missing fields."""
        instruction = f"Check for missing fields in the operation:\n{json.dumps(operation, indent=2)}\n"

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
                if "True" in model_response:
                    return GptResponseSchema(content=model_response)
                elif "False" in model_response:
                    error_message = model_response.split("False")[-1].strip()
                    return GptResponseSchema(
                        content=f"Response is False. Error: {error_message}"
                    )
                else:
                    return GptResponseSchema(content=model_response)
            else:
                return GptResponseSchema(
                    content="Sorry, an error occurred while processing operation."
                )
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return GptResponseSchema(
                content="Sorry, an error occurred while processing operation."
            )
