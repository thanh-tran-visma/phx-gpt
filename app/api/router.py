from fastapi import APIRouter, Request, Depends
from .routes import chat_endpoint
from .auth import Auth

router = APIRouter()

# Health check endpoint
@router.get("/health")
async def health_check():
    return {"status": "ok"}

# The chat route
@router.post("/chat")
async def chat_route(request: Request, token: str = Depends(Auth.get_bearer_token)):
    return await chat_endpoint(request, token)
