import logging
from sqlalchemy.orm import Session
from app.database import DatabaseManager
from app.llm import BlueViAgent, BlueViGptModel
from app.schemas import UserPromptSchema
from app.types.enum.http_status import HTTPStatus
from app.types.enum.gpt import MessageType, Role

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        db: Session,
        user_prompt: UserPromptSchema,
    ):
        self.db_manager = DatabaseManager(db)
        self.blue_vi_gpt_model = BlueViGptModel()
        self.user = user_prompt
        self.agent = BlueViAgent(
            model=self.blue_vi_gpt_model,
            db_manager=self.db_manager,
        )

    async def handle_chat(self) -> dict:
        try:
            # get user
            user = self.db_manager.get_user(self.user.uuid)
            if user is None:
                return {
                    "status": HTTPStatus.NOT_FOUND.value,
                    "response": "Invalid user id.",
                }

            # Ensure the conversation exists or create it
            conversation = self.db_manager.get_or_create_conversation(
                user.id, self.user.conversation_order
            )
            if conversation is None:
                return {
                    "status": HTTPStatus.NOT_FOUND.value,
                    "response": "Failed to create or retrieve the conversation.",
                }
            # Create UserConversation only if it does not exist
            if not (
                self.db_manager.check_user_conversation_exists(
                    user.id,
                    conversation.id,
                )
            ):
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
                self.user.prompt,
                MessageType.PROMPT,
                Role.USER,
            )
            if message is None:
                return {
                    "status": HTTPStatus.INTERNAL_SERVER_ERROR.value,
                    "response": "Failed to store the user message.",
                }

            # Let the agent handle the conversation
            bot_response = await self.agent.handle_conversation(message)

            # Store the bot's response
            self.db_manager.create_message(
                user_conversation.id,
                bot_response.content,
                MessageType.RESPONSE,
                Role.ASSISTANT,
            )

            return {
                "status": bot_response.status,
                "response": bot_response.content,
                "conversation_order": conversation.conversation_order,
                "dynamic_json": bot_response.dynamic_json,
                "type": bot_response.type
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
