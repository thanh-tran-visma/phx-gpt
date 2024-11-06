from fastapi import APIRouter, HTTPException
from app.database import DatabaseManager, Database
from app.services.end_conversation_services import EndConversationService
from app.types.enum import HTTPStatus
from app.schemas import UserPromptSchema

router = APIRouter()


@router.post("/end")
async def end_conversation_endpoint(end_conversation_data: UserPromptSchema):
    db = Database().get_session()
    db_manager = DatabaseManager(db)
    try:
        end_conversation_service = EndConversationService(db)
        end_conversation_service.handle_end_conversation(
            end_conversation_data.user_id, end_conversation_data.conversation_order
        )
        return {"message": "Conversation ended successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=f"An internal error occurred while ending the conversation: {str(e)}",
        )
    finally:
        db_manager.db.close()
