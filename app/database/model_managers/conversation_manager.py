import logging
from typing import Optional
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from app.model import Conversation, User

logger = logging.getLogger(__name__)


class ConversationManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db

    def create_conversation(self, user_id: int) -> Optional[Conversation]:
        new_conversation = Conversation(user_id=user_id)
        self.db.add(new_conversation)
        self.db.commit()
        self.db.refresh(new_conversation)
        return new_conversation

    def delete_conversation(self, user_conversation_id: int) -> bool:
        """Delete a conversation by user_conversation_id."""
        conversation = self._get_conversation_by_id(user_conversation_id)
        if conversation:
            self.db.delete(conversation)
            return self._commit_and_log(
                f"Deleted conversation with ID: {user_conversation_id}"
            )
        return False

    def _get_user_by_id(self, user_id: int) -> Optional[User]:
        """Helper to get a User by ID."""
        return self.db.query(User).filter(User.id == user_id).first()

    def _get_conversation_by_id(
        self, user_conversation_id: int
    ) -> Optional[Conversation]:
        """Helper to get a Conversation by user_conversation_id."""
        return (
            self.db.query(Conversation)
            .filter(Conversation.user_conversation_id == user_conversation_id)
            .first()
        )

    def _commit_and_log(self, success_message: str) -> bool:
        """Commit the transaction and log a success message if successful."""
        try:
            self.db.commit()
            logger.info(success_message)
            return True
        except (IntegrityError, SQLAlchemyError) as e:
            logger.error(f"Commit error: {str(e)}", exc_info=True)
            self.db.rollback()
            return False
