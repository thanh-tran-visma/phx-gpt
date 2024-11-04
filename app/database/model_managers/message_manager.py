from typing import List, Optional
from sqlalchemy.orm import Session
from app.database.helper_functions import HelperFunctions
from app.model import Message


class MessageManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db
        self.helper_functions = HelperFunctions(db)

    def create_message(
        self,
        user_id: int,
        conversation_id: int,
        content: str,
        message_type: str,
        role: str,
    ) -> Optional[Message]:
        """Create a new message."""
        new_message = Message(
            user_id=user_id,
            conversation_id=conversation_id,
            content=content,
            message_type=message_type,
            role=role,
        )
        self.db.add(new_message)
        if self.helper_functions.commit_and_log(
            f"Created message for user id {user_id} at conversation id {conversation_id}"
        ):
            return new_message
        return None

    def get_messages_by_conversation_id(
        self, conversation_id: int
    ) -> List[Message]:
        """Retrieve messages associated with a specific conversation."""
        return (
            self.db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .all()
        )

    def get_messages_by_user_conversation_id(
        self, user_id: int, conversation_id: int
    ) -> List[Message]:
        """Retrieve messages for a specific user within a specific conversation."""
        return (
            self.db.query(Message)
            .filter(
                Message.conversation_id == conversation_id,
                Message.user_id == user_id,
            )
            .all()
        )
