from typing import List, Optional
from sqlalchemy.orm import Session
from app.database.helper_functions import HelperFunctions
from app.model import Conversation


class ConversationManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db
        self.helper_functions = HelperFunctions(db)

    def create_conversation(self, user_id: int)->Optional[Conversation]:
        existing_conversation = self.db.query(Conversation).filter_by(user_id=user_id).first()
        if existing_conversation:
            return existing_conversation
        new_conversation = Conversation(user_id=user_id)
        self.db.add(new_conversation)
        self.helper_functions.commit_and_log(f"Created conversation for user id {user_id}")
        return new_conversation

    def delete_conversation(self, conversation_id: int) -> bool:
        """Delete a conversation by ID."""
        conversation = self._get_conversation_by_id(conversation_id)
        if conversation:
            self.db.delete(conversation)
            return self.helper_functions.commit_and_log(
                f"Deleted conversation with ID: {conversation_id}"
            )
        return False

    def get_conversations_by_user_id(self, user_id: int) -> List[Conversation]:
        """Retrieve all conversations for a specific user."""
        return (
            self.db.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .all()
        )

    def _get_conversation_by_id(
        self, conversation_id: int
    ) -> Optional[Conversation]:
        """Helper to get a Conversation by ID."""
        return (
            self.db.query(Conversation)
            .filter(Conversation.id == conversation_id)
            .first()
        )
