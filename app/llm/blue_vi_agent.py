import logging
from typing import List, Tuple
from app.client import PhxApiClient
from app.config import MAX_HISTORY_WINDOW_SIZE
from app.database import DatabaseManager
from app.model import Message
from app.schemas import GptResponseSchema
from app.services.cache import CacheService
from app.types.enum.gpt_response_handling import (
    BlueViResponseHandling,
)
from app.types.enum.instruction import InstructionList, CRUD

from app.types.enum.http_status import HTTPStatus
from app.utils import convert_conversation_history_to_tuples, TokenUtils


class BlueViAgent:
    def __init__(
        self, model, db_manager: DatabaseManager, cache_service: CacheService
    ):
        self.model = model
        self.token_utils = TokenUtils(self.model)
        self.db_manager = db_manager
        self.cache_service = cache_service
        self.history_window_size = MAX_HISTORY_WINDOW_SIZE
        self.phx_client = PhxApiClient()
        self.phx_client.timeout = 60

    async def get_conversation_history(
        self, message: Message
    ) -> List[tuple[str, str]]:
        """Retrieve and trim the conversation history from cache or database."""
        # First attempt to get conversation history from Redis cache
        cached_history = await self.cache_service.get_conversation_history(
            message.user_conversation_id
        )
        if cached_history:
            logging.info("Fetched conversation history from Redis cache.")
            conversation_history_list = convert_conversation_history_to_tuples(
                cached_history
            )
            for role, content in conversation_history_list:
                logging.info(
                    f"Mapped Message -> Role: {role}, Content: {content}"
                )
            return conversation_history_list
        else:
            # If not in cache, get from the database and store in Redis
            conversation_history = (
                self.db_manager.get_messages_by_user_conversation_id(
                    message.user_conversation_id
                )
            )
            logging.info("Fetched conversation history from DB.")
            conversation_history_list = convert_conversation_history_to_tuples(
                conversation_history
            )
            for role, content in conversation_history_list:
                logging.info(f"Message -> Role: {role}, Content: {content}")

            # Cache the fetched history for future use
            await self.cache_service.cache_conversation_history(
                message.user_conversation_id, conversation_history
            )
            logging.info("Fetched conversation history from DB and cached it.")
            return self.token_utils.trim_history_to_fit_tokens(
                conversation_history_list
            )

    def handle_operation_instruction(
        self,
        conversation_history: List[Tuple[str, str]],
        user_uuid: str,
        crud: CRUD,
    ) -> GptResponseSchema:
        """Handle operation-specific instructions."""
        try:
            operation_schema = self.model.assistant.handle_phx_operation(
                conversation_history, crud
            )
            operation_schema = vars(operation_schema)
            operation_schema.pop("uuid", None)

            if operation_schema:
                instruction = (
                    BlueViResponseHandling.HANDLE_OPERATION_SUCCESS.format(
                        operation='operation',
                        user_name='Alice',
                        crud=crud,
                        details=operation_schema,
                    )
                )
                response = self.model.assistant.generate_user_response_with_custom_instruction(
                    conversation_history,
                    instruction=f"{instruction}",
                )
                operation_schema["uuid"] = user_uuid
                response.dynamic_json = operation_schema
                response.operationType = InstructionList.PHX_OPERATION.value
                return response

            return GptResponseSchema(
                status=HTTPStatus.OK.value,
                response=BlueViResponseHandling.HANDLE_OPERATION_ERROR.value,
                dynamic_json=None,
            )
        except Exception as error:
            logging.error(f"Error in handle_operation_instruction: {error}")
            return GptResponseSchema(
                status=HTTPStatus.OK.value,
                response=BlueViResponseHandling.HANDLE_OPERATION_ERROR.value,
                dynamic_json=None,
            )

    def handle_general_instruction(
        self, conversation_history: List[Tuple[str, str]]
    ) -> GptResponseSchema:
        """Handle general conversation instructions."""
        try:
            return self.model.assistant.generate_user_response_with_custom_instruction(
                conversation_history=conversation_history
            )
        except Exception as error:
            logging.error(f"Error in handle_general_instruction: {error}")
            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content=f"An error occurred while processing the conversation: {error}",
                dynamic_json=None,
            )

    async def handle_conversation(
        self, user_uuid: str, message: Message
    ) -> GptResponseSchema:
        """Main entry point for handling a conversation."""
        try:
            conversation_history = await self.get_conversation_history(message)
            logging.info('conversation_history in handle_conversation')
            logging.info(conversation_history)
            decision_instruction_object = (
                self.model.assistant.identify_instruction_type(
                    conversation_history
                )
            )
            logging.info('decision_instruction_object')
            logging.info(decision_instruction_object)
            if decision_instruction_object.personal_data:
                self.db_manager.flag_message(message.id)
            if (
                decision_instruction_object.instruction.value
                == InstructionList.PHX_OPERATION.value
            ):
                return self.handle_operation_instruction(
                    conversation_history,
                    user_uuid,
                    decision_instruction_object.crud.value,
                )
            else:
                return self.handle_general_instruction(conversation_history)
        except Exception as e:
            logging.error(
                f"Unexpected error while generating chat response in agent: {e}"
            )
            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content=f"An error occurred while processing the conversation: {e}",
                dynamic_json=None,
            )
