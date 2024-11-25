import logging
import asyncio
from typing import List
from app.client import PhxApiClient
from app.config import MAX_HISTORY_WINDOW_SIZE
from app.database import DatabaseManager
from app.model import Conversation, Message
from app.schemas import GptResponseSchema
from app.services.cache import CacheService
from app.types.enum.phx_types import PhxTypes
from app.types.enum.unexpected_response_handling import (
    BlueViUnexpectedResponseHandling,
)
from app.types.enum.instruction import TrainingInstructionEnum
from app.types.enum.instruction.blue_vi_gpt_instruction_enum import (
    BlueViInstructionEnum,
)
from app.types.enum.http_status import HTTPStatus
from app.utils import TokenUtils


class BlueViAgent:
    def __init__(
        self, model, db_manager: DatabaseManager, cache_service: CacheService
    ):
        self.model = model
        self.db_manager = db_manager
        self.cache_service = cache_service
        self.token_utils = TokenUtils(self.model)
        self.history_window_size = MAX_HISTORY_WINDOW_SIZE
        self.phx_client = PhxApiClient()
        self.phx_client.timeout = 60

    async def flag_personal_data(self, prompt: str) -> bool:
        """Flag personal data in the user prompt using the assistant role."""
        return await self.model.assistant.check_for_personal_data(prompt)

    async def get_conversation_history(
        self, message: Message
    ) -> List[Conversation]:
        """Retrieve and trim the conversation history from cache or database."""
        # First attempt to get conversation history from Redis cache
        cached_history = await self.cache_service.get_conversation_history(
            message.user_conversation_id
        )
        if cached_history:
            logging.info("Fetched conversation history from Redis cache.")
            for msg in cached_history:
                if msg.id is None:
                    msg.role = "assistant"
                else:
                    msg.role = "user"
                logging.info(
                    f"Mapped Message -> Role: {msg.role}, ID: {msg.id}, Content: {msg.content}"
                )
            return cached_history
        else:
            # If not in cache, get from the database and store in Redis
            conversation_history = (
                self.db_manager.get_messages_by_user_conversation_id(
                    message.user_conversation_id
                )
            )
            logging.info('conversation_history')
            for msg in conversation_history:
                logging.info(f"Message ID: {msg.id}, Content: {msg.content}")

            # Cache the fetched history for future use
            await self.cache_service.cache_conversation_history(
                message.user_conversation_id, conversation_history
            )
            logging.info("Fetched conversation history from DB and cached it.")
            return self.token_utils.trim_history_to_fit_tokens(
                conversation_history
            )

    async def preprocess_conversation(self, message) -> List[Conversation]:
        """Preprocess the input message, flagging personal data and retrieving history."""
        try:
            # Parallel tasks for flagging and history retrieval
            tasks = [
                self.flag_personal_data(message.content),
                self.get_conversation_history(message),
            ]
            personal_data_flagged, conversation_history = await asyncio.gather(
                *tasks
            )
            logging.info('conversation_history in preprocess_conversation')
            for msg in conversation_history:
                logging.info(f"Message ID: {msg.id}, Content: {msg.content}")
            if personal_data_flagged:
                self.db_manager.flag_message(message.id)
            return conversation_history
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise

    async def handle_operation_instruction(
        self, conversation_history: list
    ) -> GptResponseSchema:
        """Handle operation-specific instructions."""
        try:
            operation_schema = await self.model.assistant.get_operation_format(
                conversation_history
            )
            if operation_schema:
                response = await self.model.generate_user_response_with_custom_instruction(
                    f"{BlueViInstructionEnum.BLUE_VI_SYSTEM_HANDLE_OPERATION_SUCCESS.value} Operation details: {operation_schema}"
                )
                response.dynamic_json = operation_schema
                response.type = PhxTypes.TOperationData.value
                return response

            return GptResponseSchema(
                status=HTTPStatus.OK.value,
                response=BlueViUnexpectedResponseHandling.HANDLE_OPERATION_ERROR.value,
                dynamic_json=None,
            )
        except Exception as error:
            logging.error(f"Error in handle_operation_instruction: {error}")
            return GptResponseSchema(
                status=HTTPStatus.OK.value,
                response=BlueViUnexpectedResponseHandling.HANDLE_OPERATION_ERROR.value,
                dynamic_json=None,
            )

    async def handle_general_instruction(
        self, conversation_history: list
    ) -> GptResponseSchema:
        """Handle general conversation instructions."""
        try:
            return await self.model.generate_user_response_with_custom_instruction(
                conversation_history
            )
        except Exception as error:
            logging.error(f"Error in handle_general_instruction: {error}")
            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content=f"An error occurred while processing the conversation: {error}",
                dynamic_json=None,
            )

    async def handle_conversation(self, message) -> GptResponseSchema:
        """Main entry point for handling a conversation."""
        try:
            conversation_history = await self.preprocess_conversation(message)
            # Identify the instruction type
            instruction_type = (
                await self.model.assistant.identify_instruction_type(
                    conversation_history
                )
            )
            logging.info(f"Instruction Type: {instruction_type}")

            # Route based on instruction type
            if (
                instruction_type
                == TrainingInstructionEnum.OPERATION_INSTRUCTION.value
            ):
                return await self.handle_operation_instruction(
                    conversation_history
                )
            else:
                return await self.handle_general_instruction(
                    conversation_history
                )

        except Exception as e:
            logging.error(
                f"Unexpected error while generating chat response: {e}"
            )
            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content=f"An error occurred while processing the conversation: {e}",
                dynamic_json=None,
            )
