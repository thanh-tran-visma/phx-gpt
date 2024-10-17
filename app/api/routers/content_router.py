from pydantic import BaseModel
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.model.models import Content
from app.schemas.content_schemas import ContentBase

router = APIRouter()

@router.get("/content", response_model=List[ContentBase])
async def get_all_content(db: Session = Depends(get_db)):
    contents = db.query(Content).all()
    return contents
