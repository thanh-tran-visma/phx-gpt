import logging
from sqlalchemy.orm import Session
from app.database import DatabaseManager
from app.schemas import UserPromptSchema
from app.types.enum import Role, MessageType, HTTPStatus

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        db: Session,
        model,
        user_prompt: UserPromptSchema,
        max_tokens: int = 2048,
        history_window_size: int = 128,
    ):
        self.db_manager = DatabaseManager(db)
        self.model = model
        self.user_prompt = user_prompt
        self.max_tokens = max_tokens
        self.history_window_size = history_window_size

    async def handle_chat(self) -> dict:
        try:
            # Create or retrieve the user
            user = self.db_manager.create_user_if_not_exists(
                self.user_prompt.user_id
            )
            if user is None:
                return {
                    "status": HTTPStatus.NOT_FOUND.value,
                    "response": "Failed to create or retrieve the user.",
                }

            # Check if conversation ID is provided; if not, create a new conversation
            if self.user_prompt.conversation_id is None:
                conversation = self.db_manager.create_conversation(user.id)
                if conversation is None:
                    return {
                        "status": HTTPStatus.NOT_FOUND.value,
                        "response": "Failed to create a conversation.",
                    }
                self.user_prompt.conversation_id = conversation.id

            # Create the user's message
            message = self.db_manager.create_message(
                user.id,
                self.user_prompt.conversation_id,
                self.user_prompt.prompt,
                MessageType.PROMPT,
                Role.USER,
            )
            if message is None:
                return {
                    "status": HTTPStatus.INTERNAL_SERVER_ERROR.value,
                    "response": "Failed to store the user message.",
                }

            # Retrieve conversation history
            conversation_history = (
                self.db_manager.get_messages_by_user_conversation_id(
                    user.id, self.user_prompt.conversation_id
                )[-self.history_window_size :]
            )

            if conversation_history:  # Ensure history is not empty
                self._trim_history_to_fit_tokens(conversation_history)

            # Get bot response
            bot_response = self.model.get_chat_response(conversation_history)

            # Store the bot's response
            self.db_manager.create_message(
                user.id,
                self.user_prompt.conversation_id,
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

    def _trim_history_to_fit_tokens(self, conversation_history: list) -> None:
        """Trim the conversation history to ensure the total token count fits within max_tokens."""
        total_tokens = sum(
            self._count_tokens(message.content)
            for message in conversation_history
        )

        while total_tokens > self.max_tokens:
            conversation_history.pop(0)  # Remove the oldest message
            total_tokens = sum(
                self._count_tokens(message.content)
                for message in conversation_history
            )

    def _count_tokens(
        self, text: str, add_bos: bool = True, special: bool = False
    ) -> int:
        """Estimate the number of tokens in a given text using the model's tokenizer."""
        if isinstance(text, str):
            text = text.encode('utf-8')

        tokens = self.model.tokenizer(text, add_bos=add_bos, special=special)
        return len(tokens)
