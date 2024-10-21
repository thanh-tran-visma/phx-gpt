from fastapi import HTTPException, Request
from app.config.env_config import BEARER_TOKEN
from app.types.enum.http_status import HTTPStatus


class Auth:
    @staticmethod
    def is_token_valid(request: Request):
        """
        Validates the token provided in the Authorization header.
        Returns True if the token is valid, otherwise raises an HTTP 401 Unauthorized error.
        """
        if not isinstance(request, Request):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value, detail="Invalid request object"
            )

        # Get the Authorization header
        authorization = request.headers.get("Authorization")
        if not authorization:
            # If Authorization header is missing, raise 401 Unauthorized
            raise HTTPException(
                status_code=HTTPStatus.UNAUTHORIZED.value,
                detail="Unauthenticated: Missing Authorization header",
            )

        # Extract the token from the "Bearer <token>" format
        if not authorization.startswith("Bearer "):
            # If it's not in the expected "Bearer <token>" format, raise 401
            raise HTTPException(
                status_code=HTTPStatus.UNAUTHORIZED.value,
                detail="Not authenticated: Invalid Authorization format",
            )

        # Extract token part
        token = authorization.split("Bearer ")[1]
        if not token:
            raise HTTPException(
                status_code=HTTPStatus.UNAUTHORIZED.value, detail="Not authenticated: Token is missing"
            )
        if not Auth.validate_token(token):
            raise HTTPException(
                status_code=HTTPStatus.UNAUTHORIZED.value, detail="Not authenticated: Invalid token"
            )

        # If token is valid, return True
        return True

    @staticmethod
    def validate_token(token: str) -> bool:
        """
        Validate the token by comparing it with the BEARER_TOKEN from the environment.
        """
        if token == BEARER_TOKEN:
            return True
        return False
