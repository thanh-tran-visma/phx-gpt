from sqlalchemy import (
    Column,
    Integer,
    Enum,
    TIMESTAMP,
    ForeignKey,
    Text,
    JSON,
)
from sqlalchemy.sql import func
from app.database.base import Base
from sqlalchemy.orm import relationship


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())

    conversations = relationship("Conversation", back_populates="user")
    messages = relationship("Message", back_populates="user")
    history_vectors = relationship("HistoryVector", back_populates="user")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())
    end_at = Column(TIMESTAMP, nullable=True)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        Integer, ForeignKey("users.id"), nullable=True
    )  # Link to user
    conversation_id = Column(
        Integer, ForeignKey("conversations.id"), nullable=False
    )
    content = Column(Text, nullable=False)
    message_type = Column(Enum("prompt", "response"), nullable=False)
    role = Column(Enum("user", "assistant"), nullable=False)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())

    user = relationship(
        "User", back_populates="messages"
    )  # Establish relationship with User
    conversation = relationship("Conversation", back_populates="messages")
    history_vectors = relationship("HistoryVector", back_populates="message")


class HistoryVector(Base):
    __tablename__ = 'history_vectors'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey('users.id'), nullable=False
    )  # Ensure this points to the correct foreign key
    conversation_id = Column(
        Integer, ForeignKey('conversations.id'), nullable=False
    )  # Link to conversation
    message_id = Column(
        Integer, ForeignKey('messages.id'), nullable=False
    )  # Link to message
    embedding_vector = Column(
        JSON, nullable=False
    )  # Using JSON type for embedding vector

    user = relationship("User", back_populates="history_vectors")
    message = relationship("Message", back_populates="history_vectors")
