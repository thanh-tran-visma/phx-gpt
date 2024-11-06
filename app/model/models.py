from sqlalchemy import (
    Column,
    Integer,
    ForeignKey,
    TIMESTAMP,
    Enum,
    Text,
    Boolean,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.database import Base


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())

    # Relationship to UserConversation
    user_conversations = relationship(
        "UserConversation", back_populates="user"
    )


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

    # Fixed the boolean type here
    sensitive_data_flag = Column(Boolean, nullable=False, default=False)
