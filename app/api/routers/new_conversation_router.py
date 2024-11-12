from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.database import Database
from app.services import NewConversationService
from app.types.enum import HTTPStatus

router = APIRouter()


class UserIdRequest(BaseModel):
    user_id: int
    uuid: str


@router.get("/new-conversation")
async def new_conversation_endpoint(request: UserIdRequest):
    database = Database()
    db = database.get_session()
    try:
        new_conversation_service = NewConversationService(db)
        response = new_conversation_service.handle_new_conversation(
            request.user_id, request.uuid
        )

        if response["status"] != HTTPStatus.OK.value:
            raise HTTPException(
                status_code=response["status"], detail=response["response"]
            )

        return {"conversation_order": response["conversation_order"]}

    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=f"An internal error occurred while creating a new conversation: {str(e)}",
        )

    finally:
        db.close()
