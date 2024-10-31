import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.types.enum.http_status import HTTPStatus


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


# Test without a token should return a 401 Unauthorized and 'detail' in the response
def test_chat_endpoint_without_token(client):
    response = client.get("/docs")
    # Assert the response status code is 401 since the token is missing
    assert response.status_code == HTTPStatus.UNAUTHORIZED.value
    json_response = response.json()
    assert "detail" in json_response
    assert (
        json_response["detail"]
        == "Unauthenticated: Missing Authorization header"
    )


@patch("app.auth.Auth.validate_token")
def test_chat_endpoint_with_mocked_token(mock_validate_token, client):
    # Simulate valid token
    mock_validate_token.return_value = True
    # Prepare the headers with a mocked token
    headers = {"Authorization": "Bearer mocked_token"}
    response = client.get("/docs", headers=headers)
    # Assert the response status code is 200 since the token is valid
    assert response.status_code == HTTPStatus.OK.value
