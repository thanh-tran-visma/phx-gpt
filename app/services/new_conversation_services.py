from sqlalchemy.orm import Session
from app.database import DatabaseManager
from app.llm import BlueViGptModel
from app.types.enum import HTTPStatus


class NewConversationService:
    def __init__(self, db: Session):
        self.db = db
        self.db_manager = DatabaseManager(db)
        self.blue_vi_gpt_model = BlueViGptModel()

    def handle_new_conversation(self, user_id: int):
        # Create and/or retrieve the user
        user = self.db_manager.create_user_if_not_exists(user_id)
        if user is None:
            return {
                "status": HTTPStatus.NOT_FOUND.value,
                "response": "User not found or could not be created.",
            }

        # Get the newest conversation
        conversation = self.db_manager.get_newest_conversation(user.id)
        return {
            "status": HTTPStatus.OK.value,
            "conversation_order": conversation.conversation_order + 1,
        }
