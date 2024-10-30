from sqlalchemy import Column, Integer, Enum, TIMESTAMP, ForeignKey, Text, BLOB
from sqlalchemy.sql import func
from app.database.base import Base
from sqlalchemy.orm import relationship


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())

    conversations = relationship("Conversation", back_populates="user")


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
    conversation_id = Column(
        Integer, ForeignKey("conversations.id"), nullable=False
    )
    content = Column(Text, nullable=False)
    message_type = Column(Enum("prompt", "response"), nullable=False)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())

    conversation = relationship("Conversation", back_populates="messages")
    vector = relationship(
        "MessageVector", back_populates="message", uselist=False
    )


class MessageVector(Base):
    __tablename__ = "message_vectors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False)
    tfidf_vector = Column(BLOB, nullable=False)
    embedding_vector = Column(BLOB, nullable=False)

    message = relationship("Message", back_populates="vector")


# Table relationships
# Ref: users.(id) - conversations.(user_id)
# Ref: conversations.(id) - messages.(conversation_id)
# Ref: messages.(id) - message_vectors.(message_id)
