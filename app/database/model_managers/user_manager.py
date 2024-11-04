from typing import Optional
from sqlalchemy.orm import Session
from app.database.helper_functions import HelperFunctions
from app.model import User


class UserManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db
        self.helper_functions = HelperFunctions(db)

    def get_user(self, user_id: int) -> Optional[User]:
        """Retrieve a user by their ID."""
        return self.db.query(User).filter(User.id == user_id).first()

    def create_user_if_not_exists(self, user_id: int) -> Optional[User]:
        """Create a user if they do not already exist, or return the existing user."""
        user = self.get_user(user_id)
        if user:
            return user
        new_user = User(id=user_id)
        self.db.add(new_user)
        if self.helper_functions.commit_and_log(
            f"Created user with ID: {user_id}"
        ):
            return new_user
        return None
