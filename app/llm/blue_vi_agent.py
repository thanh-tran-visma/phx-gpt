import logging
import asyncio
from app.client import PhxApiClient
from app.schemas import GptResponseSchema, TMethodOfConsultData
from app.types.enum.unexpected_response_handling import (
    BlueViUnexpectedResponseHandling,
)
from app.types.enum.instruction import TrainingInstructionEnum
from app.types.enum.operation import MethodOfConsultEnum
from app.types.enum.instruction.blue_vi_gpt_instruction_enum import (
    BlueViInstructionEnum,
)
from app.types.enum.http_status import HTTPStatus


class BlueViAgent:
    def __init__(self, model, db_manager, token_utils, history_window_size):
        self.model = model
        self.db_manager = db_manager
        self.token_utils = token_utils
        self.history_window_size = history_window_size
        self.phx_client = PhxApiClient()
        self.phx_client.timeout = 60

    async def flag_personal_data(self, prompt: str) -> bool:
        """Flag personal data in the user prompt using the assistant role."""
        return await self.model.assistant.check_for_personal_data(prompt)

    def get_conversation_history(self, user_conversation_id: int):
        """Retrieve and trim the conversation history."""
        conversation_history = (
            self.db_manager.get_messages_by_user_conversation_id(
                user_conversation_id
            )[-self.history_window_size :]
        )

        # Ensure we trim history to fit within token limits
        return self.token_utils.trim_history_to_fit_tokens(
            conversation_history
        )

    async def generate_response(self, conversation_history):
        """Generate a response using the user role."""
        return await self.model.user.get_chat_response(conversation_history)

    async def handle_operation_instructions(
        self, uuid: str, conversation_history: list
    ) -> GptResponseSchema:
        """Generate a response for operation instructions and check for missing fields."""
        try:
            # Fetch operation schema
            operation_schema = await self.model.assistant.get_operation_format(
                uuid, conversation_history
            )
            logging.info(f"Received operation schema: {operation_schema}")

            if operation_schema:
                # Convert methodsOfConsult if present
                if operation_schema.methodsOfConsult:
                    operation_schema.methodsOfConsult = [
                        TMethodOfConsultData(
                            shortCode=MethodOfConsultEnum(method), name=method
                        )
                        for method in operation_schema.methodsOfConsult
                    ]

            return await self.model.user.get_chat_response_with_custom_instruction(
                conversation_history,
                BlueViInstructionEnum.BLUE_VI_ASSISTANT_HANDLE_OPERATION_SUCCESS.value,
            )

        except Exception as error:
            logging.error(f"Error in handle_operation_instructions: {error}")
            return GptResponseSchema(
                status=HTTPStatus.OK.value,
                response=BlueViUnexpectedResponseHandling.HANDLE_OPERATION_ERROR.value,
            )

    async def handle_conversation(self, user, message) -> GptResponseSchema:
        """Evaluate the prompt, flag data, retrieve history, and generate a response."""
        try:
            # Ensure both coroutines are awaited correctly
            tasks = [
                self.flag_personal_data(message.content),
                asyncio.to_thread(
                    self.get_conversation_history, message.user_conversation_id
                ),  # Use to_thread for sync function
            ]

            personal_data_flagged, conversation_history = await asyncio.gather(
                *tasks
            )

            # If personal data is flagged, update the database
            if personal_data_flagged:
                self.db_manager.flag_message(message.id)

            # Identify instruction type concurrently
            instruction_type = (
                await self.model.assistant.identify_instruction_type(
                    message.content
                )
            )

            # Handle operation instruction or return a regular response
            if (
                instruction_type
                == TrainingInstructionEnum.OPERATION_INSTRUCTION.value
            ):
                return await self.handle_operation_instructions(
                    user.uuid, conversation_history
                )

            return await self.generate_response(conversation_history)

        except Exception as e:
            logging.error(
                f"Unexpected error while generating chat response: {e}"
            )
            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content=f"An error occurred while processing the conversation: {e}",
            )
