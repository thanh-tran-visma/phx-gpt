from unittest.mock import patch
import pytest

from app.types.enum.http_status import HTTPStatus


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        yield client


def test_chat_endpoint_without_token(client):
    response = client.post("bluevi-gpt/chat", json={"prompt": "test_message"})
    # Assert the response status code is 401 since the token is missing
    assert response.status_code == HTTPStatus.UNAUTHORIZED.value
    json_response = response.json()
    assert "detail" in json_response
    assert (
        json_response["detail"]
        == "Unauthenticated: Missing Authorization header"
    )


@patch("app.auth.Auth.validate_token")
@patch("app.llm.BlueViGptModel.get_chat_response")
def test_chat_endpoint_with_mocked_token(
    mock_get_chat_response, mock_validate_token, client
):
    # Simulate valid token
    mock_validate_token.return_value = True

    # Prepare the headers with a mocked token
    headers = {"Authorization": "Bearer mocked_token"}

    # Mock the response from the chat model
    mock_get_chat_response.return_value = {
        "content": "This is a mocked response."
    }

    response = client.post(
        "/bluevi-gpt/chat",
        json={"prompt": "test_message", "user_id": 1},
        headers=headers,
    )

    # Assert the response status code is 200 since the token is valid
    assert response.status_code == HTTPStatus.OK.value
    json_response = response.json()
    assert "response" in json_response
    assert json_response["response"] == "This is a mocked response."
