import logging
from typing import List

from llama_cpp import (
    Llama,
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestUserMessage,
)

from app.model import Message
from app.schemas import GptResponseSchema


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
        instruction_prompt = f"Choose the most appropriate instruction between 'Handle Operating' and 'Default' based on the context provided in: {prompt}"
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
                category = choices[0]["message"]["content"].strip()
                logging.info(f"Instruction type identified: {category}")
                return category
            return "Default"
        except Exception as e:
            logging.error(f"Error identifying instruction type: {e}")
            return "Default"

    def handle_operation_instructions(
        self, conversation_history: List[Message]
    ) -> GptResponseSchema:
        """Generate a response from the model for operation instructions."""
        try:
            # Prepare the messages for the model, ensuring the conversation history is properly mapped
            mapped_messages: List[ChatCompletionRequestUserMessage] = [
                ChatCompletionRequestUserMessage(
                    role=msg.role, content=msg.content
                )
                for msg in conversation_history
            ]
            operating_instructions = {
                "role": "user",
                "content": "Operating Instructions: Please follow the provided steps carefully to create a new operation.",
            }

            # Add the instruction to the beginning of the mapped messages
            mapped_messages.insert(0, operating_instructions)

            # Get the response from the model
            response = self.llm.create_chat_completion(
                messages=mapped_messages
            )
            choices = response.get("choices")

            if isinstance(choices, list) and len(choices) > 0:
                message_content = choices[0]["message"]["content"]
                return GptResponseSchema(content=message_content)

            return GptResponseSchema(
                content="Sorry, I couldn't generate a response."
            )
        except Exception as e:
            logging.error(
                f"Unexpected error while generating chat response: {e}"
            )
            return GptResponseSchema(
                content="Sorry, something went wrong while generating a response. Please retry with a shorter prompt."
            )
