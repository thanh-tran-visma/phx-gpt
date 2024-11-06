from fastapi import APIRouter, HTTPException
from app.database import DatabaseManager, Database
from app.types.enum import HTTPStatus
from app.schemas import UserPromptSchema

router = APIRouter()


@router.post("/end")
async def end_conversation_endpoint(end_conversation_data: UserPromptSchema):
    db = Database().get_session()
    db_manager = DatabaseManager(db)
    try:
        conversation = (
            db_manager.get_conversation_by_user_id_and_conversation_order(
                end_conversation_data.user_id,
                end_conversation_data.conversation_order,
            )
        )
        if not conversation:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="Conversation not found",
            )
        if db_manager.end_conversation(conversation.id):
            return {
                "message": "Conversation ended successfully",
                "conversation_id": conversation.id,
            }
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=f"An internal error occurred while ending the conversation: {str(e)}",
        )

    finally:
        db_manager.db.close()
