import requests
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
        # Modify or add to the data as needed for the API request
        body = {"request_data": data, "meta": {"source": "phx_gpt_client"}}
        logging.debug(f"Prepared request body: {body}")
        return body

    def get_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> requests.Response:
        """Perform a GET request."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            logging.info(f"GET request to {url} successful.")
            return response
        except requests.RequestException as e:
            logging.error(f"GET request to {url} failed: {e}")
            raise

    def post_request(
        self, endpoint: str, data: Dict[str, Any]
    ) -> requests.Response:
        """Perform a POST request."""
        url = f"{self.base_url}{endpoint}"
        body = self.prepare_body(data)
        try:
            response = requests.post(url, headers=self.headers, json=body)
            response.raise_for_status()  # Raise an exception for HTTP errors
            logging.info(f"POST request to {url} successful.")
            return response
        except requests.RequestException as e:
            logging.error(f"POST request to {url} failed: {e}")
            raise

    def delete_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> requests.Response:
        """Perform a DELETE request."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.delete(
                url, headers=self.headers, params=params
            )
            response.raise_for_status()
            logging.info(f"DELETE request to {url} successful.")
            return response
        except requests.RequestException as e:
            logging.error(f"DELETE request to {url} failed: {e}")
            raise

    def put_request(
        self, endpoint: str, data: Dict[str, Any]
    ) -> requests.Response:
        """Perform a PUT request."""
        url = f"{self.base_url}{endpoint}"
        body = self.prepare_body(data)
        try:
            response = requests.put(url, headers=self.headers, json=body)
            response.raise_for_status()
            logging.info(f"PUT request to {url} successful.")
            return response
        except requests.RequestException as e:
            logging.error(f"PUT request to {url} failed: {e}")
            raise
