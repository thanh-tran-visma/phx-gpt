from typing import Optional, List
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from app.model import Message


class MessageManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db

    def create_message(
        self,
        conversation_id: int,
        content: str,
        message_type: str,
        role: str,
        user_id: int,
    ) -> Optional[Message]:
        message = Message(
            conversation_id=conversation_id,
            content=content,
            message_type=message_type,
            role=role,
            user_id=user_id,
        )
        self.db.add(message)
        return self._commit_changes(message)

    def get_messages_by_conversation_id(
            self, conversation_id: int, user_id: int
    ) -> List[Message]:
        """Retrieve all messages associated with a given conversation ID and user ID."""
        try:
            messages = (
                self.db.query(Message)
                .filter(
                    Message.conversation_id == conversation_id,
                    Message.user_id == user_id
                )
                .order_by(Message.created_at)
                .all()
            )
            return messages
        except SQLAlchemyError:
            return []

    def _commit_changes(
        self, instance: Optional[Message] = None
    ) -> Optional[Message]:
        try:
            self.db.commit()
            if instance:
                self.db.refresh(instance)
            return instance
        except (IntegrityError, SQLAlchemyError):
            self.db.rollback()
            return None
