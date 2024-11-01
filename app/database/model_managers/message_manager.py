from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy.exc import SQLAlchemyError
from app.model import Message, MessageVector


class MessageManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db

    def create_message_with_vector(
        self,
        conversation_id: int,
        content: str,
        message_type: str,
        embedding_vector: List[float],
        role: str,
    ) -> Optional[Message]:
        """Create a new message along with its embedding vector."""
        if (
            not embedding_vector
            or not isinstance(embedding_vector, list)
            or not all(isinstance(i, float) for i in embedding_vector)
        ):
            return None

        message = Message(
            conversation_id=conversation_id,
            content=content,
            message_type=message_type,
            role=role,
        )
        self.db.add(message)
        try:
            self.db.commit()
            self.db.refresh(message)
            message_vector = MessageVector(
                message_id=message.id, embedding_vector=embedding_vector
            )
            self.db.add(message_vector)
            self.db.commit()  # Commit both message and vector together if possible
            self.db.refresh(message_vector)
            return message
        except IntegrityError:
            self.db.rollback()
            return None  # Handle integrity errors
        except SQLAlchemyError:
            self.db.rollback()
            return None  # Handle other SQLAlchemy errors

    def get_messages_by_conversation(
        self, conversation_id: int
    ) -> List[Message]:
        """Retrieve all messages for a given conversation."""
        return (
            self.db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .all()
        )
