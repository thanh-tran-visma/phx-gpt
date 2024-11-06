from typing import Optional

from sqlalchemy.orm import Session
from app.model import Conversation


class ConversationManager:
    def __init__(self, db: Session):
        self.db = db

    def get_conversation_by_order(
        self, user_id: int, conversation_order: int
    ) -> Optional[Conversation]:
        return (
            self.db.query(Conversation)
            .filter(
                Conversation.user_id == user_id,
                Conversation.conversation_order == conversation_order,
            )
            .first()
        )

    def get_or_create_conversation(
        self, user_id: int, conversation_order: int
    ) -> Optional[Conversation]:
        conversation = (
            self.db.query(Conversation)
            .filter(
                Conversation.user_id == user_id,
                Conversation.conversation_order == conversation_order,
            )
            .first()
        )
        if not conversation:
            conversation = self.create_conversation(user_id)

        # Create UserConversation if it doesn't exist
        from app.model import UserConversation

        user_conversation = (
            self.db.query(UserConversation)
            .filter(
                UserConversation.user_id == user_id,
                UserConversation.conversation_id == conversation.id,
            )
            .first()
        )
        if not user_conversation:
            user_conversation = UserConversation(
                user_id=user_id,
                conversation_id=conversation.id,
            )
            self.db.add(user_conversation)
            self.db.commit()

        return conversation

    def create_conversation(self, user_id: int) -> Optional[Conversation]:
        max_order = (
            self.db.query(Conversation.conversation_order)
            .filter(Conversation.user_id == user_id)
            .order_by(Conversation.conversation_order.desc())
            .first()
        )
        next_order = (max_order[0] + 1) if max_order else 1

        new_conversation = Conversation(
            user_id=user_id, conversation_order=next_order
        )
        self.db.add(new_conversation)
        self.db.commit()

        return new_conversation
