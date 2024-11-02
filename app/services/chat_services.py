import logging
from sqlalchemy.orm import Session
from app.database import DatabaseManager
from app.schemas import GptResponseSchema, UserPromptSchema
from app.types.enum import Role, MessageType, HTTPStatus

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, db: Session, model, user_prompt: UserPromptSchema):
        self.db_manager = DatabaseManager(db)
        self.model = model
        self.userPrompt = user_prompt

    async def handle_chat(self) -> dict:
        try:
            user = self.db_manager.create_user_if_not_exists(
                self.userPrompt.user_id
            )

            if self.userPrompt.conversation_id is None:
                conversation = self.db_manager.create_conversation(
                    user_id=user.id
                )
                if conversation is None:
                    return {
                        "status": HTTPStatus.NOT_FOUND.value,
                        "response": "Failed to create a conversation.",
                    }
                self.userPrompt.conversation_id = conversation.id

            # Create an embedding vector for the user's prompt
            user_embedding_vector = self.model.embed(self.userPrompt.prompt)
            logger.info(
                f"Embedding vector generated for prompt: {self.userPrompt.prompt}"
            )

            # Create and store the user's message
            self.db_manager.create_message_with_vector(
                conversation_id=self.userPrompt.conversation_id,
                content=self.userPrompt.prompt,
                message_type=MessageType.PROMPT,
                role=Role.USER,
                embedding_vector=user_embedding_vector,
                user_id=self.userPrompt.user_id,
            )

            # Get the conversation history
            history = self.db_manager.get_conversation_vector_history(
                self.userPrompt.conversation_id
            )

            # Prepare the bot's response based on history
            total_tokens = sum(
                len(msg.content.split()) for msg in history
            ) + len(self.userPrompt.prompt.split())
            while total_tokens > 2048 and history:
                history.pop(0)  # Maintain a maximum length for the history
                total_tokens = sum(
                    len(msg.content.split()) for msg in history
                ) + len(self.userPrompt.prompt.split())

            bot_response: GptResponseSchema = self.model.get_chat_response(
                history
            )

            # Create an embedding vector for the bot's response
            response_embedding_vector = self.model.embed(bot_response.content)

            # Create and store the assistant's message
            assistant_message = self.db_manager.create_message_with_vector(
                conversation_id=self.userPrompt.conversation_id,
                content=bot_response.content,
                message_type=MessageType.RESPONSE,
                role=Role.ASSISTANT,
                embedding_vector=response_embedding_vector,
                user_id=self.userPrompt.user_id,
            )

            # Add the assistant's response to history for embedding
            if (
                assistant_message
            ):  # Ensure the message was created successfully
                history.append(assistant_message)

            return {
                "status": HTTPStatus.OK.value,
                "response": bot_response.content,
            }

        except Exception as e:
            logger.error(f"Error in handling chat: {str(e)}")
            return {
                "status": HTTPStatus.INTERNAL_SERVER_ERROR.value,
                "response": f"An error occurred: {str(e)}",
            }
