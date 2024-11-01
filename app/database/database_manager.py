from typing import Optional, List
from sqlalchemy.orm import Session
from app.database.model_managers import (
    UserManager,
    MessageManager,
    ConversationManager,
)
from app.model import User, Message, Conversation


class DatabaseManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db
        self.user_manager = UserManager(db)
        self.message_manager = MessageManager(db)
        self.conversation_manager = ConversationManager(db)

    async def user_exists(self: Session, user_id: int) -> bool:
        return await self.query(User).filter(User.id == user_id).count() > 0

    async def create_user_if_not_exists(self, user_id: int) -> User:
        """Create a user if they do not already exist."""
        user = self.user_manager.get_user(user_id)
        if not user:
            user = await self.user_manager.create_user_if_not_exists(user_id)
        return user

    async def create_conversation(
        self, user_id: int
    ) -> Optional[Conversation]:
        return await self.conversation_manager.create_conversation(user_id)

    async def delete_conversation(self, conversation_id: int) -> None:
        self.conversation_manager.delete_conversation(conversation_id)

    async def create_message_with_vector(
        self,
        conversation_id: int,
        content: str,
        message_type: str,
        embedding_vector: List[float],
        role: str,
    ) -> Optional[Message]:
        return await self.message_manager.create_message_with_vector(
            conversation_id, content, message_type, embedding_vector, role
        )

    async def get_conversation_vector_history(self, conversation_id: int) -> List[Message]:
        return await self.message_manager.get_messages_by_conversation(conversation_id)
