import logging
from sqlalchemy.orm import Session
from fastapi import Request
from app.database import DatabaseManager  # Assuming you have this imported
from app.types.enum import Role

# Set up logging
logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, db: Session, model):
        self.db_manager = DatabaseManager(db)
        self.model = model

    async def handle_chat(self, request: Request) -> dict:
        logger.info("Handling chat request")
        body = await request.json()
        prompt = body.get("prompt", "").strip()
        user_id = body.get("user_id")
        conversation_id = body.get("conversation_id")

        if not user_id or not prompt:
            return {
                "status": "error",
                "response": "User ID or prompt not provided.",
            }

        user = self.db_manager.create_user_if_not_exists(user_id)

        if conversation_id is None:
            conversation = self.db_manager.create_conversation(user_id=user.id)
            if conversation is None:
                return {
                    "status": "error",
                    "response": "Failed to create a conversation.",
                }
            conversation_id = conversation.id

        embedding_vector = self.model.embed(prompt)

        self.db_manager.create_message_with_vector(
            conversation_id=conversation_id,
            content=prompt,
            message_type="prompt",
            role=Role.USER,
            embedding_vector=embedding_vector,
        )

        history = self.db_manager.get_conversation_vector_history(
            conversation_id, max_tokens=2048
        )
        total_tokens = sum(len(msg.content.split()) for msg in history) + len(
            prompt.split()
        )
        while total_tokens > 2048 and history:
            history.pop(0)
            total_tokens = sum(
                len(msg.content.split()) for msg in history
            ) + len(prompt.split())

        bot_response = self.model.get_chat_response(history)
        self.db_manager.create_message_with_vector(
            conversation_id=conversation_id,
            content=bot_response.content,
            message_type="response",
            role=Role.ASSISTANT,
            embedding_vector=self.model.embed(bot_response.content),
        )

        return {"status": "success", "response": bot_response.content}
