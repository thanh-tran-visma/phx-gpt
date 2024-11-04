from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from app.model import User
import logging

logger = logging.getLogger(__name__)


class UserManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db

    def get_user(self, user_id: int) -> Optional[User]:
        return self.db.query(User).filter(User.id == user_id).first()

    def create_user_if_not_exists(self, user_id: int) -> User:
        user = self.get_user(user_id)
        if user is None:
            user = User(id=user_id)
            self.db.add(user)
            self._commit_changes(user)
        return user

    def _commit_changes(
        self, instance: Optional[User] = None
    ) -> Optional[User]:
        try:
            self.db.commit()
            if instance:
                self.db.refresh(instance)
            return instance
        except (IntegrityError, SQLAlchemyError):
            self.db.rollback()
            return None
