import httpx
import logging
from typing import Any, Dict, Optional
from app.config import API_URL


class PhxApiClient:
    def __init__(self, headers: Optional[Dict[str, str]] = None):
        self.base_url = API_URL
        self.headers = headers or {"Content-Type": "application/json"}

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
        url = f"{self.base_url}{endpoint}"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url, headers=self.headers, params=params
                )
                response.raise_for_status()
                logging.info(f"GET request to {url} successful.")
                return response
        except httpx.RequestError as e:
            logging.error(f"GET request to {url} failed: {e}")
            raise

    async def post_request(
        self, endpoint: str, data: Dict[str, Any]
    ) -> httpx.Response:
        """Perform an asynchronous POST request."""
        url = f"{self.base_url}{endpoint}"
        body = self.prepare_body(data)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url, headers=self.headers, json=body
                )
                response.raise_for_status()
                logging.info(f"POST request to {url} successful.")
                return response
        except httpx.RequestError as e:
            logging.error(f"POST request to {url} failed: {e}")
            raise

    async def delete_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> httpx.Response:
        """Perform an asynchronous DELETE request."""
        url = f"{self.base_url}{endpoint}"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    url, headers=self.headers, params=params
                )
                response.raise_for_status()
                logging.info(f"DELETE request to {url} successful.")
                return response
        except httpx.RequestError as e:
            logging.error(f"DELETE request to {url} failed: {e}")
            raise

    async def put_request(
        self, endpoint: str, data: Dict[str, Any]
    ) -> httpx.Response:
        """Perform an asynchronous PUT request."""
        url = f"{self.base_url}{endpoint}"
        body = self.prepare_body(data)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    url, headers=self.headers, json=body
                )
                response.raise_for_status()
                logging.info(f"PUT request to {url} successful.")
                return response
        except httpx.RequestError as e:
            logging.error(f"PUT request to {url} failed: {e}")
            raise
