from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.database import DatabaseManager
from app.llm import BlueViGptModel
from app.types.enum import HTTPStatus


class EndConversationService:
    def __init__(self, db: Session):
        self.db = db
        self.db_manager = DatabaseManager(db)
        self.blue_vi_gpt_model = BlueViGptModel()

    def handle_end_conversation(self, user_id: int, conversation_order: int):
        # Retrieve the conversation based on user ID and order
        conversation = (
            self.db_manager.get_conversation_by_user_id_and_conversation_order(
                user_id, conversation_order
            )
        )

        if not conversation:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="Conversation not found",
            )

        user_conversation = self.db_manager.get_user_conversation(
            user_id, conversation.id
        )
        if not user_conversation:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="UserConversation not found",
            )

        # Handle sensitive messages
        self.handle_sensitive_data(user_conversation.id)

        # End the conversation and return a success response if completed
        return self.db_manager.end_conversation(conversation.id)

    def handle_sensitive_data(self, user_conversation_id: int):
        sensitive_messages = self.db_manager.get_sensitive_messages(
            user_conversation_id
        )

        # Process each sensitive message for anonymization
        for message in sensitive_messages:
            if self.blue_vi_gpt_model.check_for_personal_data(message.content):
                anonymized_message = (
                    self.blue_vi_gpt_model.get_anonymized_message(
                        message.content
                    )
                )
                self.db_manager.update_message_content(
                    message.id, anonymized_message.content
                )
        return True
