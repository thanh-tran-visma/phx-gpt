import logging
from typing import Optional, List
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from app.model import Conversation, User

logger = logging.getLogger(__name__)


class ConversationManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db

    def create_conversation(self, user_id: int) -> Optional[Conversation]:
        """Create a new conversation for the given user ID with a null embedding vector."""
        try:
            user = self._get_user_by_id(user_id)
            if not user:
                logger.warning(f"User with ID {user_id} not found.")
                return None

            conversation = Conversation(user_id=user.id)
            self.db.add(conversation)
            return self._commit_and_refresh(
                conversation,
                f"Created new conversation with ID: {conversation.id} for user ID: {user_id}",
            )
        except SQLAlchemyError as e:
            logger.error(
                f"Error creating conversation: {str(e)}", exc_info=True
            )
            self.db.rollback()
            return None

    def delete_conversation(self, conversation_id: int) -> bool:
        """Delete a conversation by ID."""
        conversation = self._get_conversation_by_id(conversation_id)
        if conversation:
            self.db.delete(conversation)
            return self._commit_and_log(
                f"Deleted conversation with ID: {conversation_id}"
            )
        return False

    def get_embedding_vector_by_conversation_id(
        self, conversation_id: int
    ) -> Optional[List[float]]:
        """Retrieve the embedding vector for a given conversation ID."""
        try:
            return (
                self.db.query(Conversation.embedding_vector)
                .filter(Conversation.id == conversation_id)
                .scalar()
            )
        except SQLAlchemyError as e:
            logger.error(
                f"Failed to retrieve embedding vector for conversation ID {conversation_id}: {str(e)}"
            )
            return None

    def update_embedding_vector(
        self, conversation_id: int, new_embedding_vector: List[float]
    ) -> Optional[Conversation]:
        """Update the embedding vector for a given conversation."""
        conversation = self._get_conversation_by_id(conversation_id)
        if conversation:
            conversation.embedding_vector = new_embedding_vector
            return self._commit_and_refresh(
                conversation,
                f"Updated embedding vector for conversation ID: {conversation_id}",
            )
        return None

    def _get_user_by_id(self, user_id: int) -> Optional[User]:
        """Helper to get a User by ID."""
        return self.db.query(User).filter(User.id == user_id).first()

    def _get_conversation_by_id(
        self, conversation_id: int
    ) -> Optional[Conversation]:
        """Helper to get a Conversation by ID."""
        return (
            self.db.query(Conversation)
            .filter(Conversation.id == conversation_id)
            .first()
        )

    def _commit_and_refresh(
        self, instance: Optional[Conversation], success_message: str
    ) -> Optional[Conversation]:
        """Commit the transaction and refresh the instance if successful."""
        try:
            self.db.commit()
            if instance:
                self.db.refresh(instance)
                logger.info(success_message)
            return instance
        except (IntegrityError, SQLAlchemyError) as e:
            logger.error(f"Commit error: {str(e)}", exc_info=True)
            self.db.rollback()
            return None

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
