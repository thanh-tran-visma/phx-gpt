import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.api.http_status import HTTPStatus

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client

# Test without a token should return a 401 Unauthorized and 'detail' in the response
def test_chat_endpoint_without_token(client):
    response = client.post("/chat/chat", json={"prompt": "test_message"})

    # Assert the response status code is 401 since the token is missing
    assert response.status_code == HTTPStatus.UNAUTHORIZED.value
    
    # Get the JSON response
    json_response = response.json()

    # Assert the response contains the 'detail' key (which indicates the error)
    assert "detail" in json_response
    assert json_response["detail"] == "Not authenticated"

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
