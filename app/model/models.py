from sqlalchemy import Column, Integer, Enum, TIMESTAMP, ForeignKey, Text
from sqlalchemy.sql import func
from database.database import Base

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    role = Column(Enum('user'), nullable=False)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, default=func.current_timestamp(), onupdate=func.current_timestamp())

class GPT(Base):
    __tablename__ = 'gpt'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    role = Column(Enum('gpt'), nullable=False)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, default=func.current_timestamp(), onupdate=func.current_timestamp())

class Content(Base):
    __tablename__ = 'content'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())

class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(Integer, nullable=False)
    sender_id = Column(Integer, ForeignKey('users.id'))
    gpt_sender_id = Column(Integer, ForeignKey('gpt.id'))
    content_id = Column(Integer, ForeignKey('content.id'))
    created_at = Column(TIMESTAMP, default=func.current_timestamp())
    end_at = Column(TIMESTAMP, nullable=True)

    # Relationships
    sender = relationship("User", backref="conversations")
    gpt_sender = relationship("GPT", backref="conversations")
    content = relationship("Content", backref="conversations")
