import logging
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.model import User

# Set up logging
logger = logging.getLogger(__name__)


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
            except IntegrityError as e:
                self.db.rollback()
                logger.error(f"IntegrityError while creating user: {e}")
                user = self.db.query(User).filter(User.id == user_id).first()
        return user

    def delete_user(self, user_id: int) -> bool:
        """Delete a user by their ID."""
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if user is None:
                logger.warning(f"User ID {user_id} not found. Cannot delete.")
                return False

            self.db.delete(user)
            self.db.commit()
            return True
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"IntegrityError while deleting user: {e}")
            return False
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting user: {e}")
            return False
