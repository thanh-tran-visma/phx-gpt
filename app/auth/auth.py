import os
from fastapi import Request, HTTPException
from app.api import HTTPStatus

class Auth:
    @staticmethod
    def get_bearer_token(request: Request):
        expected_token = os.getenv("BEARER_TOKEN")
        if expected_token is None:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="Bearer token is not configured"
            )
        
        auth_header = request.headers.get("Authorization")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=HTTPStatus.UNAUTHORIZED.value,
                detail="Not authenticated"
            )
        
        token = auth_header.split("Bearer ")[1]
        
        if token != expected_token:
            raise HTTPException(
                status_code=HTTPStatus.FORBIDDEN.value,
                detail="Not authenticated"
            )
        
        return token