from typing import List, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.model import Message, Conversation, User
from app.types.llm_user import UserPrompt


class DatabaseManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db

    def get_conversation_history(
        self, conversation_id: int
    ) -> List[UserPrompt]:
        """Retrieve conversation history from the database."""
        messages = (
            self.db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .all()
        )
        return [
            UserPrompt(role=msg.message_type, content=msg.content)
            for msg in messages
        ]

    def create_user_if_not_exists(self, user_id: int) -> User:
        """Check if the user exists; if not, create and return the user."""
        user = self.db.query(User).filter(User.id == user_id).first()
        if user is None:
            user = User(id=user_id)
            self.db.add(user)
            try:
                self.db.commit()  # Commit to save the new user
                self.db.refresh(user)  # Refresh to get the new user data
            except IntegrityError:
                self.db.rollback()  # Roll back if the user could not be added
                user = (
                    self.db.query(User).filter(User.id == user_id).first()
                )  # Try to fetch it again
        return user

    def create_conversation(self, user_id: int) -> Optional[Conversation]:
        """Create a new conversation for the given user ID."""
        try:
            user = self.create_user_if_not_exists(
                user_id
            )  # Ensure the user exists
            conversation = Conversation(user_id=user.id)
            self.db.add(conversation)
            self.db.commit()  # Commit the new conversation
            self.db.refresh(
                conversation
            )  # Refresh to get the new conversation data
            return conversation
        except IntegrityError as e:
            self.db.rollback()  # Roll back in case of an error
            print(f"IntegrityError while creating conversation: {e}")
            return None
        except Exception as e:
            self.db.rollback()  # Roll back in case of any other error
            print(f"Error creating conversation: {e}")
            return None

    def create_message(
        self, conversation_id: int, content: str, message_type: str
    ) -> Optional[Message]:
        """Create a new message."""
        try:
            message = Message(
                conversation_id=conversation_id,
                content=content,
                message_type=message_type,
            )
            self.db.add(message)
            self.db.commit()
            self.db.refresh(message)
            return message
        except Exception as e:
            self.db.rollback()
            print(f"Error creating message: {e}")
            return None
