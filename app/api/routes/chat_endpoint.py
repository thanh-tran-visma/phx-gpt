from fastapi import Request
from fastapi.responses import JSONResponse
from app.api import HTTPStatus
import time

async def chat_endpoint(request: Request):
    blue_vi_gpt_model = request.app.state.model
    try:
        body = await request.json()
        message = body.get("text", "").strip()

        if not message:
            return JSONResponse(
                status_code=HTTPStatus.BAD_REQUEST.value,
                content={"response": "No input provided."}
            )
            
        start_time = time.time()        
        grammar_correction_message = blue_vi_gpt_model.grammar_correction(message)
        anonymized_message = blue_vi_gpt_model.get_anonymized_message(message)
        bot_response = blue_vi_gpt_model.get_response(message)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return JSONResponse(
            status_code=HTTPStatus.OK.value,
            content={
                "response": bot_response,
                "execution_time": execution_time,
                "anonymized_message": anonymized_message,
                "grammar_correction_message": grammar_correction_message,
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            content={"response": f"An error occurred: {str(e)}"}
        )
