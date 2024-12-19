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
