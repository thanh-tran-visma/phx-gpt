from fastapi import APIRouter, HTTPException
from app.database import Database
from app.services import NewConversationService
from app.types.enum import HTTPStatus
from app.schemas import UserPromptSchema

router = APIRouter()


@router.get("/new-conversation")
async def new_conversation_endpoint(end_conversation_data: UserPromptSchema):
    database = Database()
    db = database.get_session()

    try:
        new_conversation_service = NewConversationService(db)
        response = new_conversation_service.handle_new_conversation(
            end_conversation_data.user_id
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
