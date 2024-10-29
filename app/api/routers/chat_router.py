from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.types.enum import HTTPStatus
import gc
from app.types.llm_types import Message

router = APIRouter()


# Chat endpoint
@router.post("/chat")
async def chat_endpoint(request: Request):
    blue_vi_gpt_model = request.app.state.model
    prompt = None

    try:
        body = await request.json()
        prompt = body.get("prompt", "").strip()

        if not prompt:
            return JSONResponse(
                status_code=HTTPStatus.BAD_REQUEST.value,
                content={"response": "No input provided."},
            )

        # Create conversation history using Message instances
        conversation_history = [Message(role="user", content=prompt)]
        bot_response = blue_vi_gpt_model.get_response(conversation_history)

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
        if prompt is not None:
            del prompt
        gc.collect()
