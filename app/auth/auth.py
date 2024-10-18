from fastapi import HTTPException, Request
from starlette.responses import JSONResponse

class Auth:
    @staticmethod
    def get_bearer_token(request: Request):
        # Get Authorization header
        authorization = request.headers.get("Authorization")
        if not authorization:
            # If Authorization header is missing, raise 401 Unauthorized
            raise HTTPException(
                status_code=401, 
                detail="Not authenticated: Missing Authorization header"
            )
        
        # Extract the token from the "Bearer <token>" format
        if not authorization.startswith("Bearer "):
            # If it's not in the expected "Bearer <token>" format, raise 401
            raise HTTPException(
                status_code=401, 
                detail="Not authenticated: Invalid Authorization format"
            )
        
        # Extract token part
        token = authorization.split("Bearer ")[1]
        if not token:
            # If token is empty, raise 401
            raise HTTPException(
                status_code=401, 
                detail="Not authenticated: Token is missing"
            )
        
        # If token is valid, return it (you could add validation logic here)
        return token
