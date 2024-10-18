from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.model.models import Gpt
from app.schemas.gpt_schemas import GptBase

router = APIRouter()

@router.get("/get-all-gpt", response_model=List[GptBase])
async def get_all_gpt(db: Session = Depends(get_db)):
    gpts = db.query(Gpt).all()
    return gpts
