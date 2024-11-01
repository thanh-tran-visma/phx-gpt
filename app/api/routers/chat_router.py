from fastapi import APIRouter, Request, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.services import ChatService
from app.types import ChatResponse
from app.types.enum import HTTPStatus

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ChatResponse}
    },
)
async def chat_endpoint(
    request: Request, db: Session = Depends(get_db)
) -> ChatResponse:
    blue_vi_gpt_model = request.app.state.model
    chat_service = ChatService(db, blue_vi_gpt_model)

    try:
        # Ensure chat_result is a string
        chat_result = await chat_service.handle_chat(request)
        if isinstance(chat_result, dict):
            # Extract response if dict was returned
            chat_result = chat_result.get(
                "response", "Error: No response found."
            )

        # Ensure we return a ChatResponse object with consistent types
        return ChatResponse(
            status=HTTPStatus.OK.value, response=str(chat_result)
        )

    except Exception as e:
        return ChatResponse(
            status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            response=f"An error occurred: {str(e)}",
        )
