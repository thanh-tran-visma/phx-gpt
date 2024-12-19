from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
import logging
from app.types.enum.http_status import HTTPStatus
from app.auth.auth import Auth
from typing import Callable


class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        try:
            Auth.is_token_valid(request)
        except HTTPException as e:
            return JSONResponse(
                content={"detail": e.detail},
                status_code=e.status_code,
            )
        except Exception as e:
            logging.error(
                f"Unexpected error occurred: {str(e)} at {request.url.path}"
            )
            return JSONResponse(
                content={"detail": "Internal Server Error"},
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            )

        response = await call_next(request)
        return response
