from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.model.models import Message
from app.schemas.message_schemas import MessageBase

router = APIRouter()

@router.get("/get-all-messages", response_model=List[MessageBase])
async def get_all_content(db: Session = Depends(get_db)):
    contents = db.query(Message).all()
    return contents
