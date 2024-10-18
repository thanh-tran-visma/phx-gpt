from fastapi import APIRouter
from app.api.routers.user_router import router as user_router
from app.api.routers.gpt_router import router as gpt_router
from app.api.routers.content_router import router as content_router
from app.api.routers.conversation_router import router as conversation_router
from app.api.routers.chat_router import router as chat_router

router = APIRouter()

# Include individual routers
router.include_router(user_router, prefix="/users", tags=["users"])
router.include_router(gpt_router, prefix="/gpt", tags=["gpt"])
router.include_router(content_router, prefix="/content", tags=["content"])
router.include_router(conversation_router, prefix="/conversations", tags=["conversations"])
router.include_router(chat_router, prefix="/chat", tags=["chat"])
