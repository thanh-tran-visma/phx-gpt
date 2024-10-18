from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import logging

from app.auth.auth import Auth

class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            token = Auth.get_bearer_token(request)
            response = await call_next(request)
            return response

        except HTTPException as e:
            if e.status_code == 401:
                logging.warning(f"Unauthorized access attempt: {request.client}")
                return JSONResponse(
                    content={"detail": "Unauthorized"},
                    status_code=401
                )
            raise e

        except Exception as e:
            logging.error(f"Unexpected error occurred: {str(e)}")
            return JSONResponse(
                content={"detail": "Internal Server Error"},
                status_code=500
            )
