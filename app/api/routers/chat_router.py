from fastapi import APIRouter, Request, HTTPException
from app.database.database import Database
from app.services import ChatService
from app.schemas import UserPromptSchema, ChatResponseSchema
from app.types.enum import HTTPStatus

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponseSchema,
    responses={
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ChatResponseSchema}
    },
)
async def chat_endpoint(
    request: Request, user_prompt: UserPromptSchema
) -> ChatResponseSchema:
    db = Database().get_session()

    try:
        blue_vi_gpt_model = request.app.state.model
        chat_service = ChatService(db, blue_vi_gpt_model, user_prompt)
        chat_result = await chat_service.handle_chat()

        if chat_result["status"] != HTTPStatus.OK.value:
            raise HTTPException(
                status_code=chat_result["status"],
                detail=chat_result["response"],
            )

        return ChatResponseSchema(
            status=HTTPStatus.OK.value, response=str(chat_result["response"])
        )

    except Exception:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="An internal error occurred.",
        )

    finally:
        db.close()
