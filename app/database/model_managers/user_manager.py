from typing import Optional
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from app.model import User


class UserManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db

    def create_user_if_not_exists(self, user_id: int) -> User:
        """Check if the user exists; if not, create and return the user."""
        user = self.db.query(User).filter(User.id == user_id).first()
        if user is None:
            user = User(id=user_id)
            self.db.add(user)
            try:
                self.db.commit()
                self.db.refresh(user)
            except IntegrityError:
                self.db.rollback()
                user = self.db.query(User).filter(User.id == user_id).first()
        return user

    def delete_user(self, user_id: int) -> bool:
        """Delete a user by their ID."""
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if user is None:
                return False

            self.db.delete(user)
            self.db.commit()
            return True
        except IntegrityError:
            self.db.rollback()
            return False
        except SQLAlchemyError:
            self.db.rollback()
            return False

    def get_user(self, user_id: int) -> Optional[User]:
        return self.db.query(User).filter(User.id == user_id).first()
