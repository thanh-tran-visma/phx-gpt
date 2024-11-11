from fastapi import APIRouter, HTTPException
from app.database.database import Database
from app.services import ChatService
from app.schemas import UserPromptSchema, ChatResponseSchema
from app.types.enum import HTTPStatus

router = APIRouter()


@router.post(
    "/chat",
    responses={
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ChatResponseSchema}
    },
)
async def chat_endpoint(user_prompt: UserPromptSchema) -> ChatResponseSchema:
    db = Database().get_session()

    try:
        chat_service = ChatService(db, user_prompt)
        chat_result = await chat_service.handle_chat()

        # Check if the status is not OK and raise an exception accordingly
        if chat_result["status"] != HTTPStatus.OK.value:
            raise HTTPException(
                status_code=chat_result["status"],
                detail=chat_result["response"],
            )

        # Ensure conversation_order is passed correctly
        return ChatResponseSchema(
            status=HTTPStatus.OK.value,
            response=str(chat_result["response"]),
            conversation_order=int(chat_result.get("conversation_order", -1)),
        )

    except Exception:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="An internal error occurred.",
        )

    finally:
        db.close()
