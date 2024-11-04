import logging
from sqlalchemy.orm import Session
from app.database import DatabaseManager
from app.schemas import GptResponseSchema, UserPromptSchema
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
        self.userPrompt = user_prompt
        self.max_tokens = max_tokens
        self.history_window_size = history_window_size

    async def handle_chat(self) -> dict:
        logger.info("Starting chat handling process.")
        try:
            # Ensure user exists or create a new user
            self.db_manager.create_user_if_not_exists(self.userPrompt.user_id)

            # Check and create conversation if not provided
            if self.userPrompt.conversation_id is None:
                conversation = self.db_manager.create_conversation(
                    self.userPrompt.user_id
                )
                if conversation is None:
                    return {
                        "status": HTTPStatus.NOT_FOUND.value,
                        "response": "Failed to create a conversation.",
                    }
                self.userPrompt.conversation_id = (
                    conversation.user_conversation_id
                )

            # Store the user message
            self.db_manager.create_message(
                conversation_id=self.userPrompt.conversation_id,
                content=self.userPrompt.prompt,
                message_type=MessageType.PROMPT,
                role=Role.USER,
                user_id=self.userPrompt.user_id,
            )

            # Retrieve conversation history
            conversation_history = (
                self.db_manager.get_messages_by_conversation_id(
                    self.userPrompt.conversation_id, self.userPrompt.user_id
                )
            )

            # Limit conversation history to the last 'history_window_size' messages
            conversation_history = conversation_history[
                -self.history_window_size :
            ]

            # Check total token count and trim if necessary
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

            # Get GPT response based on the conversation history
            bot_response: GptResponseSchema = self.model.get_chat_response(
                conversation_history
            )

            # Store the bot response message
            self.db_manager.create_message(
                conversation_id=self.userPrompt.conversation_id,
                content=bot_response.content,
                message_type=MessageType.RESPONSE,
                role=Role.ASSISTANT,
                user_id=self.userPrompt.user_id,
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

    def _count_tokens(
        self, text: str, add_bos: bool = True, special: bool = False
    ) -> int:
        """Estimate the number of tokens in a given text using the BlueViGPT tokenizer."""
        if isinstance(text, str):
            text = text.encode('utf-8')

        tokens = self.model.tokenizer(text, add_bos=add_bos, special=special)
        return len(tokens)
