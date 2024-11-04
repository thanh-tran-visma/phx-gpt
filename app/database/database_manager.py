from typing import List, Optional
from sqlalchemy.orm import Session
from app.database.model_managers import (
    UserManager,
    MessageManager,
    ConversationManager,
)
from app.model import Message, User, Conversation


class DatabaseManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db
        self.user_manager = UserManager(db)
        self.message_manager = MessageManager(db)
        self.conversation_manager = ConversationManager(db)

    # Users
    def user_exists(self, user_id: int) -> bool:
        return self.user_manager.get_user(user_id) is not None

    def create_user_if_not_exists(self, user_id: int) -> User:
        return self.user_manager.create_user_if_not_exists(user_id)

    # Conversations
    def create_conversation(self, user_id: int) -> Optional[Conversation]:
        return self.conversation_manager.create_conversation(user_id)

    def delete_conversation(self, conversation_id: int) -> None:
        self.conversation_manager.delete_conversation(conversation_id)

    def update_embedding_vector_by_conversation_id(
        self, conversation_id: int, new_embedding_vector: List[float]
    ) -> Optional[Conversation]:
        return self.conversation_manager.update_embedding_vector(
            conversation_id, new_embedding_vector
        )

    def get_embedding_vector_by_conversation_id(
        self, conversation_id: int
    ) -> Optional[List[float]]:
        return (
            self.conversation_manager.get_embedding_vector_by_conversation_id(
                conversation_id
            )
        )

    # Messages
    def create_message(
        self,
        conversation_id: int,
        content: str,
        message_type: str,
        role: str,
        user_id: int,
    ) -> Optional[Message]:
        return self.message_manager.create_message(
            conversation_id, content, message_type, role, user_id
        )

    def get_messages_by_conversation_id(
        self, conversation_id: int, user_id: int
    ) -> List[Message]:
        return self.message_manager.get_messages_by_conversation_id(
            conversation_id, user_id
        )
