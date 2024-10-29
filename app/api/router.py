from fastapi import APIRouter
from app.api.routers.chat_router import router as chat_router

router = APIRouter()
router.include_router(chat_router, prefix="/bluevi-gpt", tags=["chat"])
