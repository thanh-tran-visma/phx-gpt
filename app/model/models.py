from sqlalchemy import (
    Column,
    Integer,
    Enum as SQLAEnum,
    TIMESTAMP,
    ForeignKey,
    Text,
    func,
)
from sqlalchemy.orm import relationship
from app.database import Base
from app.types.enum import MessageType, Role


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())

    # Relationships
    conversations = relationship(
        'Conversation', back_populates='user', uselist=False
    )  # One-to-one relationship
    messages = relationship('Message', back_populates='user')


class Conversation(Base):
    __tablename__ = 'conversations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        Integer, ForeignKey('users.id'), unique=True, nullable=False
    )  # Unique user_id
    created_at = Column(TIMESTAMP, default=func.current_timestamp())
    end_at = Column(TIMESTAMP)

    # Relationships
    user = relationship(
        'User', back_populates='conversations'
    )  # One-to-one relationship
    messages = relationship('Message', back_populates='conversation')


class Message(Base):
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    conversation_id = Column(
        Integer, ForeignKey('conversations.id'), nullable=False
    )
    content = Column(Text, nullable=False)
    message_type = Column(SQLAEnum(MessageType), nullable=False)
    role = Column(SQLAEnum(Role), nullable=False)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())

    # Relationships
    user = relationship('User', back_populates='messages')
    conversation = relationship('Conversation', back_populates='messages')
