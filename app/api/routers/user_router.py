from pydantic import BaseModel
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.model.models import User
from app.schemas.user_schemas import UserBase

router = APIRouter()

@router.get("/users", response_model=List[UserBase])
async def get_all_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users
