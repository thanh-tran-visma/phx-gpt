import logging
from typing import List, Optional, Tuple, Type
from pydantic import BaseModel
from app.model import Message
from app.schemas import GptResponseSchema, PhxAppOperation, DecisionInstruction
from app.types.enum.gpt import Role
from app.types.enum.http_status import HTTPStatus
from app.types.enum.instruction import CRUD
from app.types.enum.instruction.blue_vi_gpt_instruction import (
    BlueViInstructionEnum,
)
from app.utils import (
    convert_blue_vi_response_to_schema,
    TokenUtils,
)


class BlueViGptAssistant:
    def __init__(self, llm):
        """Initialize BlueViGptAssistant with LLM and tokenizer."""
        self.llm = llm
        self.token_utils = TokenUtils(self.llm)

    def generate_user_response_with_custom_instruction(
        self,
        conversation_history: List[Tuple[str, str]],
        instruction: Optional[str] = None,
    ) -> GptResponseSchema:
        """Generate a response from the model based on conversation history for the user role, optionally with a custom instruction."""
        try:
            system_instruction = (
                instruction
                if instruction
                else BlueViInstructionEnum.BLUE_VI_SYSTEM_DEFAULT_INSTRUCTION.value
            )
            if isinstance(system_instruction, Message):
                system_instruction = system_instruction.content
            if isinstance(conversation_history, dict):
                role = Role.USER.value
                content_list = conversation_history['content']
                conversation_history = [
                    (role, str(content)) for content in content_list
                ]
            # Generate the response using the common method
            return self._create_response(
                conversation_history, system_instruction
            )

        except Exception as e:
            logging.error(
                f"Unexpected error while generating chat response in assistant: {e}"
            )
            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content="Sorry, something went wrong while generating a response.",
            )

    def get_anonymized_message(self, user_message: str) -> GptResponseSchema:
        """Anonymize the user message."""
        try:
            return self._create_response(
                [Role.USER.value, user_message],
                BlueViInstructionEnum.BLUE_VI_SYSTEM_ANONYMIZE_DATA.value,
            )

        except Exception as e:
            logging.error(f"Error anonymizing message: {e}")
            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content="Unable to anonymize the message.",
            )

    def identify_instruction_type(
        self, conversation_history: List[Tuple[str, str]]
    ) -> BaseModel:
        """Identify the type of instruction from the conversation history."""
        return self._structured_model_response(
            conversation_history,
            BlueViInstructionEnum.BLUE_VI_SYSTEM_HANDLE_INSTRUCTION_DECISION.value,
            DecisionInstruction,
        )

    def handle_phx_operation(
        self, conversation_history: List[Tuple[str, str]], crud: CRUD
    ) -> BaseModel:
        logging.info('crud in handle_phx_operation')
        logging.info(crud)
        """Generate an operation schema based on the user's conversation history and model response."""
        result = self._structured_model_response(
            conversation_history,
            BlueViInstructionEnum.BLUE_VI_SYSTEM_HANDLE_OPERATION_PROCESS.value,
            PhxAppOperation,
        )
        return result

    def _create_response(
        self, conversation_history: List[Tuple[str, str]], instruction: str
    ) -> GptResponseSchema:
        """Generate a response from the model and return a GptResponseSchema."""
        try:
            messages = [
                {"role": Role.SYSTEM.value, "content": instruction}
            ] + [
                {
                    "role": (
                        Role.USER.value
                        if sender == Role.USER.value
                        else Role.ASSISTANT.value
                    ),
                    "content": content,
                }
                for sender, content in conversation_history
            ]

            # Request the model's response
            client = self.llm["client"]
            response = client.chat.completions.create(
                model=self.llm["model"],
                messages=messages,
            )

            # Validate and extract response
            choice = next(
                (
                    c
                    for c in response.choices
                    if hasattr(c, 'message') and c.message
                ),
                None,
            )
            if choice and choice.message.content:
                return convert_blue_vi_response_to_schema(
                    choice.message.content
                )

            # Handle missing or invalid response
            error_msg = (
                "Error: Invalid choice structure or empty response text."
            )
            logging.error(error_msg)
            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content=error_msg,
            )

        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content="Error occurred while generating response.",
            )

    def _structured_model_response(
        self,
        conversation_history: List[Tuple[str, str]],
        instruction: str,
        response_format: Type[BaseModel],
    ) -> BaseModel:
        """Helper method to generate a response from the model and return a dynamically structured response."""
        messages = [{"role": Role.SYSTEM.value, "content": instruction}] + [
            {
                "role": (
                    Role.USER.value
                    if sender == Role.USER.value
                    else Role.ASSISTANT.value
                ),
                "content": content,
            }
            for sender, content in conversation_history
        ]
        logging.info('messages in _structured_model_response')
        logging.info(messages)
        client = self.llm["client"]
        response = client.beta.chat.completions.parse(
            model=self.llm["model"],
            messages=messages,
            response_format=response_format,
        )

        structured_result = response.choices[0].message.parsed
        return structured_result
