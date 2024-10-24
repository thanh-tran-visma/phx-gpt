from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.model.models import Conversation
from app.schemas.conversation_schemas import ConversationBase

router = APIRouter()


@router.get("/get-all-conversations", response_model=List[ConversationBase])
async def get_all_conversations(db: Session = Depends(get_db)):
    conversations = db.query(Conversation).all()
    return conversations
