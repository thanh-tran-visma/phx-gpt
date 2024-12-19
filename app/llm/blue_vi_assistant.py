import logging
from typing import List, Optional, Tuple, Type
from pydantic import BaseModel
from app.schemas import GptResponseSchema, PhxAppOperation, DecisionInstruction
from app.types.enum.gpt import Role
from app.types.enum.http_status import HTTPStatus
from app.types.enum.instruction import CRUD
from app.types.enum.instruction.blue_vi_gpt_instruction import (
    BlueViInstructionEnum,
)
from app.utils import convert_blue_vi_response_to_schema, TokenUtils


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
                or BlueViInstructionEnum.BLUE_VI_SYSTEM_DEFAULT_INSTRUCTION.value
            )
            if isinstance(conversation_history, dict):
                conversation_history = (
                    self._prepare_conversation_history_from_dict(
                        conversation_history
                    )
                )

            return self._create_response(
                conversation_history, system_instruction
            )
        except Exception as e:
            logging.error(
                f"Unexpected error while generating chat response in assistant: {e}"
            )
            return self._handle_error_response(
                "Sorry, something went wrong while generating a response."
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
            return self._handle_error_response(
                "Unable to anonymize the message."
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
        """Generate an operation schema based on the user's conversation history and model response."""
        logging.info(f'CRUD operation in handle_phx_operation: {crud}')
        return self._structured_model_response(
            conversation_history,
            BlueViInstructionEnum.BLUE_VI_SYSTEM_HANDLE_OPERATION_PROCESS.value,
            PhxAppOperation,
        )

    def _create_response(
        self, conversation_history: List[Tuple[str, str]], instruction: str
    ) -> GptResponseSchema:
        """Generate a response from the model and return a GptResponseSchema."""
        try:
            messages = self._prepare_messages(
                conversation_history, instruction
            )
            client = self.llm["client"]
            response = client.chat.completions.create(
                model=self.llm["model"], messages=messages
            )

            return self._process_model_response(response)
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return self._handle_error_response(
                "Error occurred while generating response."
            )

    def _structured_model_response(
        self,
        conversation_history: List[Tuple[str, str]],
        instruction: str,
        response_format: Type[BaseModel],
    ) -> BaseModel:
        """Helper method to generate a response from the model and return a dynamically structured response."""
        messages = self._prepare_messages(conversation_history, instruction)
        client = self.llm["client"]
        response = client.beta.chat.completions.parse(
            model=self.llm["model"],
            messages=messages,
            response_format=response_format,
        )
        return response.choices[0].message.parsed

    @staticmethod
    def _prepare_messages(
        conversation_history: List[Tuple[str, str]], instruction: str
    ) -> List[dict]:
        """Prepare the messages list for the model request."""
        return [{"role": Role.SYSTEM.value, "content": instruction}] + [
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

    @staticmethod
    def _prepare_conversation_history_from_dict(
        conversation_history: dict,
    ) -> List[Tuple[str, str]]:
        """Prepare conversation history from dictionary structure."""
        role = Role.USER.value
        content_list = conversation_history.get('content', [])
        return [(role, str(content)) for content in content_list]

    def _process_model_response(self, response) -> GptResponseSchema:
        """Process the model's response and return it as a GptResponseSchema."""
        choice = next(
            (
                c
                for c in response.choices
                if hasattr(c, 'message') and c.message
            ),
            None,
        )
        if choice and choice.message.content:
            return convert_blue_vi_response_to_schema(choice.message.content)

        error_msg = "Error: Invalid choice structure or empty response text."
        logging.error(error_msg)
        return self._handle_error_response(error_msg)

    @staticmethod
    def _handle_error_response(error_msg: str) -> GptResponseSchema:
        """Return a standardized error response."""
        return GptResponseSchema(
            status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            content=error_msg,
        )
