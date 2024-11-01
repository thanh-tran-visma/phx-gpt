from typing import Optional
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from app.model import Conversation, User


class ConversationManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db

    def create_conversation(self, user_id: int) -> Optional[Conversation]:
        """Create a new conversation for the given user ID."""
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if user is None:
                return None

            conversation = Conversation(user_id=user.id)
            self.db.add(conversation)
            self.db.commit()
            self.db.refresh(conversation)
            return conversation
        except IntegrityError:
            self.db.rollback()
            return None
        except SQLAlchemyError:
            self.db.rollback()
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
                return None  # Return None if the conversation doesn't exist
        except IntegrityError:
            self.db.rollback()
        except SQLAlchemyError:
            self.db.rollback()
            return None  # Handle other SQLAlchemy errors
