from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import logging
from app.types.enum import HTTPStatus
from app.auth.auth import Auth

class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip token validation for docs, redoc, and openapi.json
        if request.url.path.startswith("/docs") or request.url.path.startswith("/redoc") or request.url.path.startswith("/openapi.json"):
            response = await call_next(request)
            return response
        
        try:
            # Validate token
            if not Auth.is_token_valid(request):
                raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED.value, detail="Unauthorized access")
            
            # Proceed with the request
            response = await call_next(request)
            return response

        except HTTPException as e:
            if e.status_code == HTTPStatus.UNAUTHORIZED.value:
                logging.warning(f"Unauthorized access attempt: {request.client}")
                return JSONResponse(
                    content={"detail": "Unauthorized access"},
                    status_code=HTTPStatus.UNAUTHORIZED.value
                )
            raise e

        except Exception as e:
            logging.error(f"Unexpected error occurred: {str(e)}")
            return JSONResponse(
                content={"detail": "Internal Server Error"},
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value
            )
