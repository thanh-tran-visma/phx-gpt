from fastapi import APIRouter, Request, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from starlette.responses import JSONResponse
import logging

from app.database import get_db
from app.services import ChatService
from app.types.enum import HTTPStatus

router = APIRouter()

# Set up logging
logger = logging.getLogger(__name__)


# Define Pydantic models for request and response
class ChatResponse(BaseModel):
    status: str
    response: str


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": dict}},
)
async def chat_endpoint(
    request: Request, db: Session = Depends(get_db)
) -> ChatResponse | JSONResponse:
    blue_vi_gpt_model = request.app.state.model
    chat_service = ChatService(db, blue_vi_gpt_model)

    try:
        chat_result = await chat_service.handle_chat(request)

        # Ensure that chat_result contains a valid string for response
        if isinstance(chat_result, dict):
            return ChatResponse(
                status=chat_result.get("status"),
                response=chat_result.get("response"),
            )

        return ChatResponse(status="success", response=chat_result)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            content={"response": f"An error occurred: {str(e)}"},
        )
