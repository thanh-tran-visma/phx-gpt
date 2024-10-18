import logging
from starlette.middleware.cors import CORSMiddleware
from app.config.env_config import API_URL

class CORSConfig:
    def __init__(self, app):
        self.app = app
        self._origins = API_URL

    def add_cors_middleware(self):
        """
        Adds the CORS middleware to the app with the specified settings.
        """
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self._origins,
            allow_credentials=True, 
            allow_methods=["*"], 
            allow_headers=["*"],
        )

    def set_origins(self, origins):
        """
        Dynamically set the allowed origins. 
        """
        # Optionally, validate origins before applying them
        self._origins = self._validate_origins(origins)
        self.add_cors_middleware()

    def get_origins(self):
        """
        Get the current CORS origins.
        """
        return self._origins

    def _validate_origins(self, origins):
        """
        Validate and sanitize origins before setting them.
        This is where you can implement checks (e.g., ensuring they are valid URLs).
        """
        validated_origins = []
        for origin in origins:
            if self._is_valid_origin(origin):
                validated_origins.append(origin)
            else:
                logging.warning(f"Invalid CORS origin: {origin}")
        return validated_origins

    def _is_valid_origin(self, origin):
        # Implement a simple check or regex to validate origin format
        return origin.startswith("http://") or origin.startswith("https://")
