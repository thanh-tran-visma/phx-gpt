from sqlalchemy.orm import Session
from fastapi import Request
from app.database import DatabaseManager
from app.types import GptResponse, UserPrompt
from app.types.enum import Role, MessageType, HTTPStatus


class ChatService:
    def __init__(self, db: Session, model):
        self.db_manager = DatabaseManager(db)
        self.model = model
        self.userPrompt = UserPrompt(prompt='', user_id=-1, conversation_id=-1)

    async def handle_chat(self, request: Request) -> dict:
        try:
            body = await request.json()
            self.userPrompt.prompt = body.get("prompt", "").strip()
            self.userPrompt.user_id = body.get("user_id")
            self.userPrompt.conversation_id = body.get("conversation_id")

            if not self.userPrompt.user_id or not self.userPrompt.prompt:
                return {
                    "status": HTTPStatus.NOT_FOUND.value,
                    "response": "User ID or prompt not provided.",
                }

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

            embedding_vector = self.model.embed(self.userPrompt.prompt)

            self.db_manager.create_message_with_vector(
                conversation_id=self.userPrompt.conversation_id,
                content=self.userPrompt.prompt,
                message_type=MessageType.PROMPT,
                role=Role.USER,
                embedding_vector=embedding_vector,
            )

            history = self.db_manager.get_conversation_vector_history(
                self.userPrompt.conversation_id, max_tokens=2048
            )
            total_tokens = sum(
                len(msg.content.split()) for msg in history
            ) + len(self.userPrompt.prompt.split())
            while total_tokens > 2048 and history:
                history.pop(0)
                total_tokens = sum(
                    len(msg.content.split()) for msg in history
                ) + len(self.userPrompt.prompt.split())

            bot_response: GptResponse = self.model.get_chat_response(history)
            response_embedding_vector = self.model.embed(bot_response.content)

            self.db_manager.create_message_with_vector(
                conversation_id=self.userPrompt.conversation_id,
                content=bot_response.content,
                message_type=MessageType.RESPONSE,
                role=Role.ASSISTANT,
                embedding_vector=response_embedding_vector,
            )
            return {
                "status": HTTPStatus.OK.value,
                "response": bot_response.content,
            }

        except Exception as e:
            return {
                "status": HTTPStatus.INTERNAL_SERVER_ERROR.value,
                "response": f"An error occurred: {str(e)}",
            }
