from fastapi import APIRouter, HTTPException
from app.database.database import Database
from app.services.routes import ChatService
from app.schemas import UserPromptSchema, ChatResponseSchema
from app.types.enum.http_status import HTTPStatus
from fastapi import Request

router = APIRouter()


@router.post(
    "/chat",
    responses={
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ChatResponseSchema}
    },
)
async def chat_endpoint(
    user_prompt: UserPromptSchema,
    request: Request,
) -> ChatResponseSchema:
    database = Database()
    db = database.get_session()

    try:
        chat_service = ChatService(db, user_prompt, request)
        chat_result = await chat_service.handle_chat()

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
            dynamic_json=chat_result.get("dynamic_json"),
            time_taken=float(chat_result.get("time_taken", 0.0)),
        )

    except Exception:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="An internal error occurred",
        )

    finally:
        db.close()
