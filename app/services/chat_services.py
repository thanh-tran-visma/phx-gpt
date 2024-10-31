import logging
from fastapi import Request
from sqlalchemy.orm import Session
from app.database import DatabaseManager

# Set up logging
logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, db: Session, model) -> None:
        self.db = db
        self.model = model
        self.db_manager = DatabaseManager(db)

    async def handle_chat(self, request: Request) -> dict:
        body = await request.json()
        prompt = body.get("prompt", "").strip()
        user_id = body.get("user_id")
        conversation_id = body.get("conversation_id")

        if not user_id or not prompt:
            return {
                "status": "error",
                "response": "User ID or prompt not provided.",
            }

        # Create or retrieve conversation
        if conversation_id is None:
            conversation = self.db_manager.create_conversation(user_id=user_id)
            if conversation is None:
                return {
                    "status": "error",
                    "response": "Failed to create a conversation.",
                }
            conversation_id = conversation.id

        # Generate embedding for the current prompt
        embedding_vector = self.model.get_embedding(prompt)

        # Log the embedding vector for debugging
        logger.info(embedding_vector)

        if embedding_vector is None:
            self.db_manager.delete_conversation(conversation_id)
            return {
                "status": "error",
                "response": "Failed to generate a valid embedding. Conversation deleted.",
            }

        # Create the message with its vector
        self.db_manager.create_message_with_vector(
            conversation_id=conversation_id,
            content=prompt,
            message_type="prompt",
            embedding_vector=embedding_vector,
        )

        # Retrieve message history
        history = self.db_manager.get_conversation_vector_history(
            conversation_id, max_tokens=2048
        )
        logger.info(history)

        total_tokens = sum(len(msg.content.split()) for msg in history) + len(
            prompt.split()
        )
        while total_tokens > 2048 and history:
            history.pop(0)
            total_tokens = sum(
                len(msg.content.split()) for msg in history
            ) + len(prompt.split())

        # Get bot response
        bot_response = self.model.get_chat_response(history)

        # Save bot response message
        self.db_manager.create_message_with_vector(
            conversation_id=conversation_id,
            content=bot_response.content,
            message_type="response",
            embedding_vector=(
                bot_response.embedding
                if hasattr(bot_response, 'embedding')
                else None
            ),
        )

        return {"status": "success", "response": bot_response.content}
