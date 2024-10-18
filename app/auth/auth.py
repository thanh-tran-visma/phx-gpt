import os
from fastapi import Request, HTTPException
from app.api import HTTPStatus

class Auth:
    @staticmethod
    def get_bearer_token(request: Request):
        expected_token = os.getenv("BEARER_TOKEN")
        
        # Ensure the expected token exists
        if expected_token is None:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="Bearer token is not configured"
            )
        
        auth_header = request.headers.get("Authorization")
        
        # If Authorization header is missing or not a Bearer token, raise 401 Unauthorized
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=HTTPStatus.UNAUTHORIZED.value,
                detail="Not authenticated"
            )
        
        token = auth_header.split("Bearer ")[1]
        
        # If token does not match the expected token, raise 403 Forbidden
        if token != expected_token:
            raise HTTPException(
                status_code=HTTPStatus.FORBIDDEN.value,
                detail="Not authenticated"
            )
        
        return token
