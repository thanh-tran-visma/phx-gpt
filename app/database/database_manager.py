from typing import List, Optional
from sqlalchemy.orm import Session

from app.database.model_managers import (
    UserManager,
    MessageManager,
    ConversationManager,
    HistoryVectorManager,
)
from app.model import Message, User, Conversation


class DatabaseManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db
        self.user_manager = UserManager(db)
        self.message_manager = MessageManager(db)
        self.conversation_manager = ConversationManager(db)
        self.history_vector_manager = HistoryVectorManager(db)

    def user_exists(self: Session, user_id: int) -> bool:
        return self.query(User).filter(User.id == user_id).count() > 0

    def create_user_if_not_exists(self, user_id: int) -> User:
        """Create a user if they do not already exist."""
        return self.user_manager.create_user_if_not_exists(user_id)

    def create_conversation(self, user_id: int) -> Optional[Conversation]:
        """Create a new conversation for the given user."""
        return self.conversation_manager.create_conversation(user_id)

    def delete_conversation(self, conversation_id: int) -> None:
        """Delete a conversation by ID."""
        self.conversation_manager.delete_conversation(conversation_id)

    def create_message_with_vector(
        self,
        conversation_id: int,
        content: str,
        message_type: str,
        embedding_vector: List[float],
        role: str,
        user_id: int,
    ) -> Optional[Message]:
        """Create a new message along with its embedding vector and store user_id."""
        return self.message_manager.create_message_with_vector(
            conversation_id,
            content,
            message_type,
            embedding_vector,
            role,
            user_id,
        )

    def get_conversation_vector_history(
        self, conversation_id: int
    ) -> List[Message]:
        """Retrieve message history for the given conversation."""
        return self.message_manager.get_messages_by_conversation(
            conversation_id
        )

    def create_history_vector(
        self,
        user_id: int,
        conversation_id: int,
        message_id: int,
        embedding_vector: List[float],
    ) -> None:
        """Create a history vector for a given message."""
        self.history_vector_manager.create_history_vector(
            user_id, conversation_id, message_id, embedding_vector
        )
