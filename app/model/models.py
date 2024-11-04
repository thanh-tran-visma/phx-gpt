from sqlalchemy import Column, Integer, Enum, TIMESTAMP, ForeignKey, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database.base import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())

    conversations = relationship("Conversation", back_populates="user")
    messages = relationship("Message", back_populates="user")


class Conversation(Base):
    __tablename__ = "conversations"

    user_conversation_id = Column(
        Integer, primary_key=True, autoincrement=True
    )  # Make this the primary key
    user_id = Column(
        Integer, ForeignKey("users.id"), nullable=False
    )  # Foreign key only
    created_at = Column(TIMESTAMP, default=func.current_timestamp())
    end_at = Column(TIMESTAMP, nullable=True)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user_conversation_id = Column(
        Integer,
        ForeignKey("conversations.user_conversation_id"),
        nullable=False,
    )
    content = Column(Text, nullable=False)
    message_type = Column(Enum("prompt", "response"), nullable=False)
    role = Column(Enum("user", "assistant"), nullable=False)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())

    user = relationship("User", back_populates="messages")
    conversation = relationship("Conversation", back_populates="messages")
