import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.api.http_status import HTTPStatus

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client

# Mock the get_bearer_token function to simulate a valid token
@patch("app.auth.auth.Auth.get_bearer_token", return_value="mocked_token")
def test_chat_endpoint_with_mocked_token(client, mock_auth):
    # Make the request without including the token in the headers
    response = client.post("/chat/chat", json={"prompt": "test_message"})

    # Assert the response status code is 200 (OK) since the token is mocked
    assert response.status_code == HTTPStatus.OK.value
    
    # Get the JSON response
    json_response = response.json()

    # Assert the presence of expected keys in the response
    assert "response" in json_response
    assert isinstance(json_response["response"], str)
    assert json_response["response"] != ""

@patch("os.getenv", return_value="mocked_token")
@patch("app.auth.auth.Auth.get_bearer_token", return_value="mocked_token")
def test_chat_endpoint_with_token(mock_getenv, mock_auth, client):
    # Set the mocked Bearer token in the headers
    headers = {"Authorization": "Bearer mocked_token"}
    response = client.post("/chat/chat", json={"prompt": "test_message"}, headers=headers)
    
    # Assert the response status code is 200 (OK) since the token is provided
    assert response.status_code == HTTPStatus.OK.value
    
    json_response = response.json()
    assert "response" in json_response
    assert isinstance(json_response["response"], str)
    assert json_response["response"] != ""
