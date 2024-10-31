import logging
from typing import Optional, List

from sqlalchemy.orm import Session

from app.database.model_managers import (
    UserManager,
    MessageManager,
    ConversationManager,
)
from app.model import User, Message, Conversation

# Set up logging
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db
        self.user_manager = UserManager(db)
        self.message_manager = MessageManager(db)
        self.conversation_manager = ConversationManager(db)

    def create_user_if_not_exists(self, user_id: int) -> User:
        """Create a user if they do not already exist."""
        user = self.user_manager.get_user(user_id)
        if not user:
            user = self.user_manager.create_user_if_not_exists(user_id)
            logger.info(f"User created: {user_id}")
        else:
            logger.info(f"User already exists: {user_id}")
        return user

    def create_conversation(self, user_id: int) -> Optional[Conversation]:
        return self.conversation_manager.create_conversation(user_id)

    def delete_conversation(self, conversation_id: int) -> None:
        self.conversation_manager.delete_conversation(conversation_id)

    def create_message_with_vector(
        self,
        conversation_id: int,
        content: str,
        message_type: str,
        embedding_vector: List[float],
        role: str,
    ) -> Optional[Message]:
        return self.message_manager.create_message_with_vector(
            conversation_id, content, message_type, embedding_vector, role
        )

    def get_conversation_vector_history(
        self, conversation_id: int, max_tokens: int
    ) -> List[Message]:
        return self.message_manager.get_messages_by_conversation(
            conversation_id
        )

    def user_exists(self: Session, user_id: int) -> bool:
        return self.query(User).filter(User.id == user_id).count() > 0
