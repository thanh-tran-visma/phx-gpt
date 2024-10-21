from sqlalchemy import Column, Integer, Enum, TIMESTAMP, ForeignKey, Text
from sqlalchemy.sql import func
from app.database.base import Base
from sqlalchemy.orm import relationship


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())
    end_at = Column(TIMESTAMP, nullable=True)

    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    content = Column(Text, nullable=False)
    message_type = Column(Enum("prompt", "response"), nullable=False)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())

    conversation = relationship("Conversation", back_populates="messages")
