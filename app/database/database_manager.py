from typing import List, Optional
from sqlalchemy.orm import Session
from app.database.model_managers import (
    UserManager,
    MessageManager,
    ConversationManager,
    UserConversationManager,
)
from app.model import Message, Conversation, User, UserConversation


class DatabaseManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db
        self.user_manager = UserManager(db)
        self.message_manager = MessageManager(db)
        self.conversation_manager = ConversationManager(db)
        self.user_conversations_manager = UserConversationManager(db)

    # Users
    def user_exists(self, user_id: int) -> bool:
        return self.user_manager.get_user(user_id) is not None

    def create_user_if_not_exists(self, user_id: int) -> Optional[User]:
        return self.user_manager.create_user_if_not_exists(user_id)

    # Conversations
    def get_conversation_by_order(
        self, user_id: int, conversation_order: int
    ) -> Optional[Conversation]:
        return self.conversation_manager.get_conversation_by_order(
            user_id, conversation_order
        )

    def create_conversation(self, user_id: int) -> Optional[Conversation]:
        return self.conversation_manager.create_conversation(user_id)

    def get_or_create_conversation(
        self, user_id: int, conversation_order: Optional[int] = None
    ) -> Optional[Conversation]:
        return self.conversation_manager.get_or_create_conversation(
            user_id, conversation_order
        )

    # Messages
    def create_message(
        self,
        user_conversation_id: int,
        content: str,
        message_type: str,
        role: str,
    ) -> Optional[Message]:
        return self.message_manager.create_message(
            user_conversation_id, content, message_type, role
        )

    def get_messages_by_user_conversation_id(
        self, user_conversation_id: int
    ) -> List[Message]:
        return self.message_manager.get_messages_by_user_conversation_id(
            user_conversation_id
        )

    # User Conversations
    def get_user_conversation(
        self, user_id: int, conversation_id: int
    ) -> Optional[UserConversation]:
        return self.user_conversations_manager.get_user_conversation(
            user_id, conversation_id
        )

    def get_conversations_for_user(
        self, user_id: int
    ) -> List[UserConversation]:
        return self.user_conversations_manager.get_conversations_for_user(
            user_id
        )

    def create_user_conversation(
        self, user_id: int, conversation_id: int, conversation_order: int
    ) -> Optional[UserConversation]:
        return self.user_conversations_manager.create_user_conversation(
            user_id, conversation_id, conversation_order
        )

    def check_user_conversation_exists(
        self, user_id: int, conversation_id: int
    ) -> bool:
        return self.user_conversations_manager.check_user_conversation_exists(
            user_id, conversation_id
        )
