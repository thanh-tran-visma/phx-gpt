from fastapi import APIRouter

from app.api.routers.new_conversation_router import (
    router as new_conversation_router,
)
from app.api.routers.chat_router import router as chat_router
from app.api.routers.end_conversation_router import (
    router as end_conversation_router,
)
from app.types.enum import HTTPStatus


router = APIRouter()
router.include_router(chat_router, prefix="/bluevi-gpt")
router.include_router(new_conversation_router, prefix="/bluevi-gpt")
router.include_router(end_conversation_router, prefix="/bluevi-gpt")


@router.get("/auth")
async def auth_check():
    return {"status": HTTPStatus.OK.value}
