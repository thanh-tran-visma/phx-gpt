import logging
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.model import Conversation, User

# Set up logging
logger = logging.getLogger(__name__)


class ConversationManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db

    def create_conversation(self, user_id: int) -> Optional[Conversation]:
        """Create a new conversation for the given user ID."""
        try:
            user = (
                self.db.query(User).filter(User.id == user_id).first()
            )  # Check if user exists in the database
            if user is None:
                logger.warning(
                    f"User ID {user_id} not found. Cannot create conversation."
                )
                return None  # Return None if user doesn't exist

            conversation = Conversation(user_id=user.id)
            self.db.add(conversation)
            self.db.commit()
            self.db.refresh(conversation)
            return conversation
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"IntegrityError while creating conversation: {e}")
            return None
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating conversation: {e}")
            return None

    def delete_conversation(self, conversation_id: int) -> None:
        """Delete a conversation by ID."""
        try:
            conversation = (
                self.db.query(Conversation)
                .filter(Conversation.id == conversation_id)
                .first()
            )
            if conversation:
                self.db.delete(conversation)
                self.db.commit()
            else:
                logger.warning(
                    f"Attempted to delete non-existing conversation ID: {conversation_id}"
                )
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"IntegrityError while deleting conversation: {e}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting conversation: {e}")
