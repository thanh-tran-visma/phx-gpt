from typing import Optional, List

from sqlalchemy.orm import Session
from app.model import Message, UserConversation


class MessageManager:
    def __init__(self, db: Session):
        self.db = db

    def get_messages_for_conversation(
        self, user_conversation_id: int
    ) -> List[Message]:
        return (
            self.db.query(Message)
            .filter(Message.user_conversation_id == user_conversation_id)
            .all()
        )

    def create_message(
        self,
        user_conversation_id: int,
        content: str,
        message_type: str,
        role: str,
    ) -> Optional[Message]:
        new_message = Message(
            user_conversation_id=user_conversation_id,
            content=content,
            message_type=message_type,
            role=role,
        )
        self.db.add(new_message)
        self.db.commit()
        self.db.refresh(new_message)
        return new_message

    def get_messages_by_user_conversation_id(
        self, user_conversation_id: int
    ) -> List[Message]:
        return (
            self.db.query(Message)
            .filter(Message.user_conversation_id == user_conversation_id)
            .all()
        )

    def get_messages_by_conversation_id(
        self, conversation_id: int
    ) -> List[Message]:
        return (
            self.db.query(Message)
            .join(UserConversation)
            .filter(UserConversation.conversation_id == conversation_id)
            .all()
        )
