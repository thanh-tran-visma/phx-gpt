import logging

import httpx

from app.config import API_URL
from typing import Dict, Any, Optional

from app.client.base_client import BaseClient


class PhxApiClient(BaseClient):
    def __init__(self, headers: Optional[Dict[str, str]] = None):
        super().__init__(base_url=API_URL, headers=headers)

    @staticmethod
    def prepare_body(data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the request body with provided data."""
        body = {"request_data": data, "meta": {"source": "phx_gpt_client"}}
        logging.debug(f"Prepared request body: {body}")
        return body

    async def get_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> httpx.Response:
        """Perform an asynchronous GET request."""
        return await self._request("GET", endpoint, params=params)

    async def post_request(
        self, endpoint: str, data: Dict[str, Any]
    ) -> httpx.Response:
        """Perform an asynchronous POST request."""
        body = self.prepare_body(data)
        return await self._request("POST", endpoint, data=body)

    async def delete_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> httpx.Response:
        """Perform an asynchronous DELETE request."""
        return await self._request("DELETE", endpoint, params=params)

    async def put_request(
        self, endpoint: str, data: Dict[str, Any]
    ) -> httpx.Response:
        """Perform an asynchronous PUT request."""
        body = self.prepare_body(data)
        return await self._request("PUT", endpoint, data=body)
