from sqlalchemy import (
    Column,
    Integer,
    ForeignKey,
    TIMESTAMP,
    Enum,
    Text,
    Boolean,
    String,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.database import Base
from sqlalchemy.inspection import inspect


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(
        String(36),
        unique=True,
        nullable=False,
        default=None,
    )
    created_at = Column(TIMESTAMP, default=func.current_timestamp())

    # Relationship to UserConversation
    user_conversations = relationship(
        "UserConversation", back_populates="user"
    )

    def to_dict(self, depth=1):
        """
        Serialize the User model instance into a dictionary.
        """
        return serialize_model(self, depth)


class UserConversation(Base):
    __tablename__ = 'user_conversations'

    id = Column(Integer, primary_key=True, autoincrement=True)

    user_id = Column(
        Integer, ForeignKey('users.id'), nullable=False
    )  # ForeignKey to User
    conversation_id = Column(
        Integer, ForeignKey('conversations.id'), nullable=False
    )  # ForeignKey to Conversation

    user = relationship("User", back_populates="user_conversations")
    conversation = relationship(
        "Conversation", back_populates="user_conversations"
    )
    messages = relationship("Message", back_populates="user_conversation")

    def to_dict(self, depth=1):
        """
        Serialize the UserConversation model instance into a dictionary.
        """
        return serialize_model(self, depth)


class Conversation(Base):
    __tablename__ = 'conversations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    conversation_order = Column(Integer, index=True)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())
    end_at = Column(TIMESTAMP, nullable=True)

    user_conversations = relationship(
        'UserConversation', back_populates='conversation'
    )

    def to_dict(self, depth=1):
        """
        Serialize the Conversation model instance into a dictionary.
        """
        return serialize_model(self, depth)


class Message(Base):
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_conversation_id = Column(
        Integer, ForeignKey('user_conversations.id'), nullable=False
    )
    content = Column(Text, nullable=False)
    message_type = Column(Enum('prompt', 'response'), nullable=False)
    role = Column(Enum('user', 'assistant'), nullable=False)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())

    user_conversation = relationship(
        'UserConversation', back_populates='messages'
    )

    sensitive_data_flag = Column(Boolean, nullable=False, default=False)

    def to_dict(self, depth=1):
        """
        Serialize the Message model instance into a dictionary.
        """
        return serialize_model(self, depth)


# Serializer utility
def serialize_model(instance, depth=1):
    """
    Convert an SQLAlchemy model instance to a dictionary, optionally including related objects.
    """
    if not instance:
        return {}

    # Initialize a dictionary for the model's attributes
    serialized = {
        column.key: getattr(instance, column.key)
        for column in inspect(instance).mapper.columns
    }

    # If depth > 0, serialize related models
    if depth > 0:
        for rel in instance.__mapper__.relationships:
            related_obj = getattr(instance, rel)
            if related_obj:
                if isinstance(related_obj, list):
                    serialized[rel] = [
                        serialize_model(rel_instance, depth - 1)
                        for rel_instance in related_obj
                    ]
                else:
                    serialized[rel] = serialize_model(related_obj, depth - 1)

    return serialized
