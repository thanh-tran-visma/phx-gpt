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
    def get_user(self, user_id: int) -> Optional[User]:
        return self.user_manager.get_user(user_id)

    def create_user_if_not_exists(
        self, user_id: int, uuid: str
    ) -> Optional[User]:
        return self.user_manager.create_user_if_not_exists(user_id, uuid)

    # Conversations
    def get_conversation_by_user_id_and_conversation_order(
        self, user_id: int, conversation_order: int
    ) -> Optional[Conversation]:
        return self.conversation_manager.get_conversation_by_user_id_and_conversation_order(
            user_id, conversation_order
        )

    def create_conversation(self, user_id: int) -> Optional[Conversation]:
        return self.conversation_manager.create_conversation(user_id)

    def end_conversation(self, conversation_id: int) -> bool:
        return self.conversation_manager.end_conversation(conversation_id)

    def get_or_create_conversation(
        self, user_id: int, conversation_order: Optional[int] = None
    ) -> Optional[Conversation]:
        return self.conversation_manager.get_or_create_conversation(
            user_id, conversation_order
        )

    def get_newest_conversation(
        self, user_id: int, conversation_order: Optional[int] = None
    ) -> Optional[Conversation]:
        return self.conversation_manager.get_newest_conversation(
            user_id, conversation_order
        )

    def get_conversations_by_user_id(self, user_id: int):
        return self.conversation_manager.get_conversation_by_user_id(user_id)

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

    def get_sensitive_messages(
        self, user_conversation_id: int
    ) -> List[Message]:
        return self.message_manager.get_sensitive_messages(
            user_conversation_id
        )

    def update_message_content(self, message_id: int, content: str):
        return self.message_manager.update_message_content(message_id, content)

    def flag_message(self, message_id: int):
        return self.message_manager.flag_message(message_id)

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
        self, user_id: int, conversation_id: int
    ) -> Optional[UserConversation]:
        return self.user_conversations_manager.create_user_conversation(
            user_id, conversation_id
        )

    def check_user_conversation_exists(
        self, user_id: int, conversation_id: int
    ) -> bool:
        return self.user_conversations_manager.check_user_conversation_exists(
            user_id, conversation_id
        )
