import logging
from typing import Optional, List

from sqlalchemy.orm import Session

from app.database.model_managers import UserManager, MessageManager, ConversationManager
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
        return self.user_manager.create_user_if_not_exists(user_id)

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
    ) -> Optional[Message]:
        return self.message_manager.create_message_with_vector(
            conversation_id, content, message_type, embedding_vector
        )

    def get_conversation_vector_history(
        self, conversation_id: int, max_tokens: int
    ) -> List[Message]:
        # Implement this method to retrieve the history of messages in a conversation
        return self.message_manager.get_messages_by_conversation(
            conversation_id
        )  # Adjust as necessary
