from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
import logging
from app.types.enum import HTTPStatus
from app.auth.auth import Auth
from typing import Callable


class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        # Skip token validation for docs, redoc, openapi.json, and health checks
        if request.url.path.startswith(
            ("/docs", "/redoc", "/openapi.json", "/health")
        ):
            return await call_next(request)

        # Validate token
        try:
            Auth.is_token_valid(request)
        except HTTPException as e:
            logging.warning(
                f"Unauthorized access attempt from {request.client} for {request.url.path}: {e.detail}"
            )
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

        # Proceed with the request if token is valid
        response = await call_next(request)
        return response
