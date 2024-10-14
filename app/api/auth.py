import os
from fastapi import Request, HTTPException
from app.api import HTTPStatus
class Auth:
    def get_bearer_token(request: Request):
        expected_token = os.getenv("BEARER_TOKEN")

        # Get the Authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED.value, detail="Authorization header missing or incorrect")

        # Extract the token from the Authorization header
        token = auth_header.split("Bearer ")[1]

        # Validate the token
        if token != expected_token:
            raise HTTPException(status_code=HTTPStatus.FORBIDDEN.value, detail="Invalid Bearer token")

        return token
