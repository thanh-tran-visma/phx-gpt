from typing import List
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.model import HistoryVector


class HistoryVectorManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db

    def create_history_vector(
        self,
        user_id: int,
        conversation_id: int,
        message_id: int,
        embedding_vector: List[float],
    ) -> None:
        """Create a new history vector."""
        history_vector = HistoryVector(
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            embedding_vector=embedding_vector,
        )
        self.db.add(history_vector)
        try:
            self.db.commit()
            self.db.refresh(history_vector)
        except IntegrityError:
            self.db.rollback()
        except SQLAlchemyError:
            self.db.rollback()
