from typing import List, Optional

from sqlalchemy.orm import Session
from app.model import UserConversation


class UserConversationManager:
    def __init__(self, db: Session):
        self.db = db

    def get_user_conversation(
        self, user_id: int, conversation_id: int
    ) -> UserConversation:
        return (
            self.db.query(UserConversation)
            .filter(
                UserConversation.user_id == user_id,
                UserConversation.conversation_id == conversation_id,
            )
            .first()
        )

    def get_conversations_for_user(
        self, user_id: int
    ) -> List[UserConversation]:
        return (
            self.db.query(UserConversation)
            .filter(UserConversation.user_id == user_id)
            .all()
        )

    def create_user_conversation(
        self, user_id: int, conversation_id: int
    ) -> Optional[UserConversation]:
        user_conversation = UserConversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        self.db.add(user_conversation)
        self.db.commit()
        return user_conversation

    def check_user_conversation_exists(
        self, user_id: int, conversation_id: int
    ) -> bool:
        return (
            self.db.query(UserConversation)
            .filter(
                UserConversation.user_id == user_id,
                UserConversation.conversation_id == conversation_id,
            )
            .count()
            > 0
        )
