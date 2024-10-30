import logging
from typing import List, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.model import Message, Conversation, User, MessageVector
from app.types.llm_user import UserPrompt

# Set up logging
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, db: Session) -> None:
        self.db: Session = db

    def get_conversation_history(self, conversation_id: int) -> List[UserPrompt]:
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
            user = self.create_user_if_not_exists(user_id)  # Ensure the user exists
            conversation = Conversation(user_id=user.id)
            self.db.add(conversation)
            self.db.commit()  # Commit the new conversation
            self.db.refresh(conversation)  # Refresh to get the new conversation data
            return conversation
        except IntegrityError as e:
            self.db.rollback()  # Roll back in case of an error
            logger.error(f"IntegrityError while creating conversation: {e}")
            return None
        except Exception as e:
            self.db.rollback()  # Roll back in case of any other error
            logger.error(f"Error creating conversation: {e}")
            return None

    def create_message_with_vector(
            self, conversation_id: int, content: str, message_type: str, embedding_vector: List[float]
    ) -> Optional[Message]:
        """Create a new message along with its embedding vector."""
        try:
            # Log the embedding vector to ensure it's valid
            logger.info(
                f"Creating message with vector. Content: {content}, Message Type: {message_type}, Embedding Vector: {embedding_vector}")

            # Ensure the embedding_vector is not None and is a list of floats
            if embedding_vector is None or not isinstance(embedding_vector, list) or not all(
                    isinstance(i, float) for i in embedding_vector):
                logger.error("Invalid embedding vector provided: must be a non-null list of floats.")
                return None

            message = Message(
                conversation_id=conversation_id,
                content=content,
                message_type=message_type,
            )
            self.db.add(message)
            self.db.commit()
            self.db.refresh(message)

            # Store the embedding vector directly
            message_vector = MessageVector(
                message_id=message.id,
                embedding_vector=embedding_vector  # Store directly as a list of floats
            )
            self.db.add(message_vector)
            self.db.commit()
            self.db.refresh(message_vector)

            return message
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating message with vector: {e}")
            return None

    def get_messages_by_conversation(self, conversation_id: int) -> List[Message]:
        """Retrieve all messages for a given conversation."""
        return self.db.query(Message).filter(Message.conversation_id == conversation_id).all()

    def get_message_vector(self, message_id: int) -> Optional[MessageVector]:
        """Retrieve the vector associated with a given message."""
        return self.db.query(MessageVector).filter(MessageVector.message_id == message_id).first()

    def get_message_vectors_by_conversation(self, conversation_id: int) -> List[MessageVector]:
        """Retrieve all message vectors for a given conversation."""
        return self.db.query(MessageVector).filter(MessageVector.conversation_id == conversation_id).all()

    def get_conversation_vector_history(self, conversation_id: int, max_tokens: int = 2048) -> List[UserPrompt]:
        """Retrieve conversation history from the database in vector format."""
        messages = self.get_messages_by_conversation(conversation_id)  # Use existing method

        # Create a list to hold the formatted messages
        user_prompts = [UserPrompt(role=msg.message_type, content=msg.content) for msg in messages]

        # Check total tokens and truncate if necessary
        total_tokens = sum(len(msg.content.split()) for msg in user_prompts)

        if total_tokens > max_tokens:
            logger.warning(f"Truncating conversation history to fit context window. Total tokens: {total_tokens}")
            # Implement truncation logic (keeping only the last messages)
            while total_tokens > max_tokens and user_prompts:
                user_prompts.pop(0)  # Remove the oldest message
                total_tokens = sum(len(msg.content.split()) for msg in user_prompts)

        return user_prompts
