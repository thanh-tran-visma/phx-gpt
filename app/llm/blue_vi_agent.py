import logging
from typing import List

from app.model import Message, User
from app.schemas import GptResponseSchema
from app.types.enum.http_status import HTTPStatus
from app.types.enum.instruction import InstructionEnum


class BlueViAgent:
    def __init__(self, model, db_manager, token_utils, history_window_size):
        self.model = model
        self.db_manager = db_manager
        self.token_utils = token_utils
        self.history_window_size = history_window_size

    async def flag_personal_data(self, prompt: str) -> bool:
        """Flag personal data in the user prompt using the assistant role."""
        is_personal_data = await self.model.assistant.check_for_personal_data(
            prompt
        )
        if is_personal_data:
            logging.warning(f"Personal data detected: {prompt}")
        return is_personal_data

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

    async def generate_response(
        self, conversation_history
    ) -> GptResponseSchema:
        """Generate a response using the user role."""
        return await self.model.user.get_chat_response(conversation_history)

    async def handle_operation_instructions(
        self, uuid: str, conversation_history: List[Message]
    ) -> GptResponseSchema:
        """Generate a response for operation instructions and check for missing fields."""
        try:
            # Call to get_operation_format
            operation_schema = await self.model.assistant.get_operation_format(
                uuid, conversation_history
            )
            logging.info(f"Received operation schema: {operation_schema}")

            # Validate schema before proceeding
            if operation_schema and any(
                [
                    getattr(operation_schema, field) is not None
                    for field in operation_schema.__annotations__.keys()
                ]
            ):
                logging.info("Valid operation schema received:")
                logging.info(operation_schema)
            else:
                logging.warning(
                    "Operation schema fields are empty or missing."
                )

            # Return user response
            return await self.model.user.get_chat_response(
                conversation_history
            )

        except Exception as e:
            logging.error(f"Error in handle_operation_instructions: {e}")
            raise

    async def handle_conversation(
        self, user: User, message: Message
    ) -> GptResponseSchema:
        """Evaluate the prompt, flag data, retrieve history, and generate a response."""
        try:
            # Flag personal data if detected
            if await self.flag_personal_data(message.content):
                self.db_manager.flag_message(message.id)

            # Get trimmed conversation history
            conversation_history = self.get_conversation_history(
                message.user_conversation_id
            )

            # Identify if this prompt needs special handling
            instruction_type = (
                await self.model.assistant.identify_instruction_type(
                    message.content
                )
            )
            logging.info('instruction_type: %s', instruction_type)
            if instruction_type == InstructionEnum.OPERATION_INSTRUCTION.value:
                return await self.handle_operation_instructions(
                    user.uuid, conversation_history
                )
            # If no special instruction, generate the standard response
            return await self.generate_response(conversation_history)

        except Exception as e:
            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content=f"An error occurred while processing the conversation:{e}",
            )
