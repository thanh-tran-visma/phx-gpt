from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional
from app.types.enum import HTTPStatus
from app.types.llm_user import UserPrompt
from app.database import DatabaseManager, get_db

router = APIRouter()


@router.post("/chat")
async def chat_endpoint(
    request: Request, db: Session = Depends(get_db)
) -> JSONResponse:
    blue_vi_gpt_model = request.app.state.model
    db_manager = DatabaseManager(db)
    prompt: Optional[str] = None
    user_id: Optional[int] = None
    conversation_id: Optional[int] = None

    try:
        body = await request.json()
        prompt = body.get("prompt", "").strip()
        user_id = body.get("user_id")
        conversation_id = body.get("conversation_id")

        if not user_id or not prompt:
            return JSONResponse(
                status_code=HTTPStatus.BAD_REQUEST.value,
                content={"response": "User ID or prompt not provided."},
            )

        # Check if a conversation ID is provided, else create a new one
        if conversation_id is None:
            conversation = db_manager.create_conversation(user_id=user_id)
            if conversation is None:
                return JSONResponse(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                    content={"response": "Failed to create a conversation."},
                )
            conversation_id = conversation.id

        # Store the user message in the database
        db_manager.create_message(
            conversation_id, prompt, message_type="prompt"
        )

        # Retrieve conversation history
        conversation_history = db_manager.get_conversation_history(
            conversation_id
        )
        conversation_history.append(UserPrompt(role="user", content=prompt))

        # Generate bot response using LLM
        bot_response = blue_vi_gpt_model.get_response(conversation_history)

        # Store the bot message in the database
        db_manager.create_message(
            conversation_id, bot_response.content, message_type="response"
        )

        return JSONResponse(
            status_code=HTTPStatus.OK.value,
            content={"response": bot_response.content},
        )

    except Exception as e:
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            content={"response": f"An error occurred: {str(e)}"},
        )

    finally:
        # Clear the prompt variable if it was set
        if prompt is not None:
            del prompt
