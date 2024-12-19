from typing import Optional
from sqlalchemy.orm import Session

from app.model import User


class UserManager:
    def __init__(self, db: Session):
        self.db = db

    def create_user(self) -> User:
        new_user = User()
        self.db.add(new_user)
        self.db.commit()
        return new_user

    def create_user_if_not_exists(
        self, uuid: str, user_name: str
    ) -> Optional[User]:
        user = self.get_user(uuid)
        if user is None:
            user = User(uuid=uuid, user_name=user_name)
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
        return user

    def get_user(self, uuid: str) -> Optional[User]:
        return self.db.query(User).filter(User.uuid == uuid).first()
