from fastapi import APIRouter
from app.api.routers.message_router import router as message_router
from app.api.routers.conversation_router import router as conversation_router
from app.api.routers.chat_router import router as chat_router

router = APIRouter()

router.include_router(message_router, prefix="/messages", tags=["messages"])
router.include_router(
    conversation_router, prefix="/conversations", tags=["conversations"]
)
router.include_router(chat_router, prefix="/chat", tags=["chat"])
