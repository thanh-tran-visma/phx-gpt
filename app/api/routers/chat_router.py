import logging
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional
from app.types.enum import HTTPStatus
from app.database import DatabaseManager, get_db

# Set up logging
logging.basicConfig(level=logging.INFO)  # You can change the level to DEBUG for more detailed logs
logger = logging.getLogger(__name__)

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
    embedding_vector: List[float] = None

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

        if conversation_id is None:
            conversation = db_manager.create_conversation(user_id=user_id)
            if conversation is None:
                return JSONResponse(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                    content={"response": "Failed to create a conversation."},
                )
            conversation_id = conversation.id

        # Generate embedding for the current prompt
        embedding_vector = blue_vi_gpt_model.get_embedding(prompt)

        # Log the embedding vector for debugging
        logger.info(f"Embedding vector for prompt '{prompt}': {embedding_vector}")

        # Store the user message along with its embedding vector
        db_manager.create_message_with_vector(
            conversation_id=conversation_id,
            content=prompt,
            message_type="prompt",
            embedding_vector=embedding_vector
        )

        # Load conversation history, considering context window size
        history = db_manager.get_conversation_vector_history(conversation_id, max_tokens=2048)
        logger.info(history)
        # Check if the combined total tokens exceed the maximum limit (4096)
        total_tokens = sum(len(msg.content.split()) for msg in history) + len(prompt.split())

        if total_tokens > 4096:
            logger.warning(f"Total token count exceeds limit: {total_tokens}. Adjusting history.")
            # Further truncate history to fit within 4096 tokens
            while total_tokens > 4096 and history:
                history.pop(0)  # Remove the oldest message
                total_tokens = sum(len(msg.content.split()) for msg in history) + len(prompt.split())

        # Generate bot response using the adjusted conversation history
        bot_response = blue_vi_gpt_model.get_chat_response(history)

        # Log the bot response for debugging
        logger.info(f"Bot response: {bot_response.content}")

        # Store the bot message in the database
        db_manager.create_message_with_vector(
            conversation_id=conversation_id,
            content=bot_response.content,
            message_type="response",
            embedding_vector=bot_response.embedding if hasattr(bot_response, 'embedding') else None
        )

        return JSONResponse(
            status_code=HTTPStatus.OK.value,
            content={"response": bot_response.content},
        )

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")  # Log the error
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            content={"response": f"An error occurred: {str(e)}"},
        )

    finally:
        if prompt is not None:
            del prompt
