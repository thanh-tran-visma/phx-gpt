from fastapi import Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from app.api import HTTPStatus
import gc
import os

async def chat_endpoint(request: Request):
    blue_vi_gpt_model = request.app.state.model
    prompt = None

    # Retrieve Bearer token from environment
    expected_token = os.getenv("BEARER_TOKEN")

    # Get the Authorization header
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or incorrect")

    # Extract the token from the Authorization header
    token = auth_header.split("Bearer ")[1]

    # Validate the token
    if token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid Bearer token")

    try:
        body = await request.json()
        prompt = body.get("prompt", "").strip()

        if not prompt:
            return JSONResponse(
                status_code=HTTPStatus.BAD_REQUEST.value,
                content={"response": "No input provided."}
            )

        # Grammar correction and response generation
        grammar_correction_message = blue_vi_gpt_model.grammar_correction(prompt)
        bot_response = blue_vi_gpt_model.get_response(grammar_correction_message)

        return JSONResponse(
            status_code=HTTPStatus.OK.value,
            content={"response": bot_response}
        )

    except Exception as e:
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            content={"response": f"An error occurred: {str(e)}"}
        )
    
    finally:
        if prompt is not None:
            del prompt
        gc.collect()