import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.llm.llm_model import BlueViGptModel
from app.api.http_status import HTTPStatus

@pytest.fixture(scope="module")
def client():
    app.state.model = BlueViGptModel()
    app.state.model.load_model()
    with TestClient(app) as client:
        yield client

def test_chat_endpoint_without_token(client):
    response = client.post("/chat/chat", json={"prompt": "test_message"})

    # Assert the response status code is 401 (Unauthorized) because no token is provided
    assert response.status_code == HTTPStatus.UNAUTHORIZED.value

    # Get the JSON response
    json_response = response.json()

    # Assert the presence of expected keys in the response
    assert "detail" in json_response
    assert json_response["detail"] == "Not authenticated"

def test_chat_endpoint_with_token(client):
    # Set the Bearer token in the headers
    headers = {"Authorization": "Bearer 1234"}
    response = client.post("/chat/chat", json={"prompt": "test_message"}, headers=headers)

    # Assert the response status code is 200 (OK) since the token is provided
    assert response.status_code == HTTPStatus.OK.value

    # Get the JSON response
    json_response = response.json()

    # Assert the presence of expected keys in the response
    assert "response" in json_response
    assert isinstance(json_response["response"], str)
    assert json_response["response"] != ""
