from fastapi import APIRouter, Request
from .routes import chat_endpoint

router = APIRouter()

# Health check endpoint
@router.get("/health")
async def health_check():
    return {"status": "ok"}

# The chat route
@router.post("/chat")
async def chat_route(request: Request):
    return await chat_endpoint(request)
