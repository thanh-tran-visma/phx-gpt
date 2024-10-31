from typing import List, Optional
from sqlalchemy.orm import Session
from app.model import MessageVector


class VectorManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db

    def get_message_vector(self, message_id: int) -> Optional[MessageVector]:
        """Retrieve the vector associated with a given message."""
        return (
            self.db.query(MessageVector)
            .filter(MessageVector.message_id == message_id)
            .first()
        )

    def get_message_vectors_by_conversation(
        self, conversation_id: int
    ) -> List[MessageVector]:
        """Retrieve all message vectors for a given conversation."""
        return (
            self.db.query(MessageVector)
            .filter(MessageVector.conversation_id == conversation_id)
            .all()
        )
