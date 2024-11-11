import logging
from sqlalchemy.orm import Session
from app.database import DatabaseManager
from app.llm import Agent, BlueViGptModel
from app.schemas import UserPromptSchema
from app.types.enum import Role, MessageType, HTTPStatus
from app.utils import TokenUtils
from app.config.config_env import LLM_MAX_TOKEN, MAX_HISTORY_WINDOW_SIZE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        db: Session,
        user_prompt: UserPromptSchema,
        max_tokens=LLM_MAX_TOKEN,
        history_window_size=MAX_HISTORY_WINDOW_SIZE,
    ):
        self.db_manager = DatabaseManager(db)
        self.blue_vi_gpt_model = BlueViGptModel()
        self.user_prompt = user_prompt
        self.history_window_size = history_window_size
        self.token_utils = TokenUtils(self.blue_vi_gpt_model, max_tokens)
        self.agent = Agent(
            model=self.blue_vi_gpt_model,
            db_manager=self.db_manager,
            token_utils=self.token_utils,
            history_window_size=self.history_window_size,
        )

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

            # Log conversation order for debugging purposes
            logger.info(
                f"Conversation order: {conversation.conversation_order}"
            )

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
                        "conversation_order": -1,
                    }
            else:
                # If it already exists, retrieve the existing user conversation
                user_conversation = self.db_manager.get_user_conversation(
                    user.id, conversation.id
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

            # Let the agent handle the conversation
            bot_response = self.agent.handle_conversation(message)

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
                "conversation_order": conversation.conversation_order,
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
