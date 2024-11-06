from sqlalchemy.orm import Session
from app.database import DatabaseManager
from app.schemas import UserPromptSchema
from app.types.enum import Role, MessageType, HTTPStatus
from app.utils import TokenUtils
from app.config.config_env import LLM_MAX_TOKEN, MAX_HISTORY_WINDOW_SIZE
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        db: Session,
        model,
        user_prompt: UserPromptSchema,
        max_tokens=LLM_MAX_TOKEN,
        history_window_size=MAX_HISTORY_WINDOW_SIZE,
    ):
        self.db_manager = DatabaseManager(db)
        self.model = model
        self.user_prompt = user_prompt
        self.history_window_size = history_window_size
        self.token_utils = TokenUtils(model, max_tokens)

    async def handle_chat(self) -> dict:
        try:
            # Create and/or retrieve the user
            user = self.db_manager.create_user_if_not_exists(
                self.user_prompt.user_id
            )
            if user is None:
                return {
                    "status": HTTPStatus.NOT_FOUND.value,
                    "response": "Failed to create or retrieve the user.",
                }

            # Ensure the conversation exists or create it
            conversation = self.db_manager.get_or_create_conversation(
                user.id, self.user_prompt.conversation_order
            )
            if conversation is None:
                return {
                    "status": HTTPStatus.NOT_FOUND.value,
                    "response": "Failed to create or retrieve the conversation.",
                }

            # Check if the UserConversation already exists
            user_conversation_exists = (
                self.db_manager.check_user_conversation_exists(
                    user.id,
                    conversation.id,
                )
            )

            # Create UserConversation only if it does not exist
            if not user_conversation_exists:
                user_conversation = self.db_manager.create_user_conversation(
                    user.id,
                    conversation.id,
                )
                if user_conversation is None:
                    return {
                        "status": HTTPStatus.INTERNAL_SERVER_ERROR.value,
                        "response": "Failed to create user conversation.",
                    }
            else:
                # If it already exists, retrieve the existing user conversation
                user_conversation = self.db_manager.get_user_conversation(
                    user.id, conversation.id
                )

            # Flag user prompt for personal data
            is_personal_data = self.model.check_for_personal_data(
                self.user_prompt.prompt
            )

            # Ensure is_personal_data is a boolean and log the result
            if is_personal_data:
                logger.warning(
                    f"Personal data detected in user prompt: {self.user_prompt.prompt}"
                )
            else:
                logger.info(
                    f"No personal data detected in user prompt: {self.user_prompt.prompt}"
                )

            # Create the user's message
            message = self.db_manager.create_message(
                user_conversation.id,
                self.user_prompt.prompt,
                MessageType.PROMPT,
                Role.USER,
            )
            if message is None:
                return {
                    "status": HTTPStatus.INTERNAL_SERVER_ERROR.value,
                    "response": "Failed to store the user message.",
                }

            # Retrieve and trim conversation history
            conversation_history = (
                self.db_manager.get_messages_by_user_conversation_id(
                    user_conversation.id
                )[-self.history_window_size :]
            )
            trimmed_history = self.token_utils.trim_history_to_fit_tokens(
                conversation_history
            )

            # Get bot response
            bot_response = self.model.get_chat_response(trimmed_history)

            # Store the bot's response
            self.db_manager.create_message(
                user_conversation.id,
                bot_response.content,
                MessageType.RESPONSE,
                Role.ASSISTANT,
            )

            return {
                "status": HTTPStatus.OK.value,
                "response": bot_response.content,
            }

        except Exception as e:
            logger.error(
                f"An error occurred during chat handling: {str(e)}",
                exc_info=True,
            )
            return {
                "status": HTTPStatus.INTERNAL_SERVER_ERROR.value,
                "response": "An error occurred while processing your request.",
            }
