from typing import List, Optional
from sqlalchemy.orm import Session
from app.database.model_managers import (
    UserManager,
    MessageManager,
    ConversationManager,
)
from app.model import Message, Conversation, User


class DatabaseManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db
        self.user_manager = UserManager(db)
        self.message_manager = MessageManager(db)
        self.conversation_manager = ConversationManager(db)

    # Users
    def user_exists(self, user_id: int) -> bool:
        """Check if a user exists by their ID."""
        return self.user_manager.get_user(user_id) is not None

    def create_user_if_not_exists(self, user_id: int) -> Optional[User]:
        """Create a user if they do not already exist, or return the existing user."""
        return self.user_manager.create_user_if_not_exists(user_id)

    # Conversations
    def get_conversations_by_user_id(self, user_id: int) -> List[Conversation]:
        """Retrieve all conversations for a specific user."""
        return self.conversation_manager.get_conversations_by_user_id(user_id)

    def create_conversation(self, user_id: int) -> Optional[Conversation]:
        """Create a new conversation for the given user ID."""
        return self.conversation_manager.create_conversation(user_id)

    def delete_conversation(self, conversation_id: int) -> bool:
        """Delete a conversation by its ID."""
        return self.conversation_manager.delete_conversation(conversation_id)

    # Messages
    def create_message(
        self,
        user_id: int,
        conversation_id: int,
        content: str,
        message_type: str,
        role: str,
    ) -> Optional[Message]:
        """Create a new message associated with a user conversation."""
        return self.message_manager.create_message(
            user_id, conversation_id, content, message_type, role
        )

    def get_messages_by_conversation_id(
        self, conversation_id: int
    ) -> List[Message]:
        """Retrieve messages associated with a specific conversation."""
        return self.message_manager.get_messages_by_conversation_id(
            conversation_id
        )

    def get_messages_by_user_conversation_id(
        self, user_id: int, conversation_id: int
    ) -> List[Message]:
        """Retrieve messages associated with a specific user and conversation."""
        return self.message_manager.get_messages_by_user_conversation_id(
            user_id, conversation_id
        )
