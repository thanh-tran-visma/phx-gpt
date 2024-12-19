import httpx
import logging
from typing import Any, Dict, Optional


class BaseClient:
    def __init__(
        self, base_url: str, headers: Optional[Dict[str, str]] = None
    ):
        self.base_url = base_url
        self.headers = headers or {"Content-Type": "application/json"}

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Generalized HTTP request handler for different HTTP methods."""
        url = f"{self.base_url}/{endpoint}"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method, url, headers=self.headers, json=data, params=params
                )
                response.raise_for_status()
                return response
        except httpx.RequestError as e:
            logging.error(f"{method} request to {url} failed: {e}")
            raise
