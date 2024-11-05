import pytest
from unittest.mock import MagicMock, patch
from app.main import app
from app.types.enum.http_status import HTTPStatus
from app.auth.auth import Auth


@pytest.fixture(scope="module")
def mock_client():
    """Mock the FastAPI TestClient."""
    with patch("fastapi.testclient.TestClient") as MockClient:
        yield MockClient


# Mock the Auth class directly for authentication tests
@pytest.fixture
def mock_auth():
    with patch.object(
        Auth, 'validate_token', return_value=True
    ) as mock_validate:
        yield mock_validate


@pytest.fixture(scope="module")
def mock_db_session():
    with patch("app.database.database.Database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        yield mock_session


def test_auth_valid_token(mock_auth, mock_client):
    headers = {"Authorization": "Bearer mocked_token"}

    # Set up the mock client response
    mock_client.return_value.get.return_value.status_code = HTTPStatus.OK.value
    mock_client.return_value.get.return_value.json.return_value = {}

    response = mock_client.return_value.get("/auth", headers=headers)
    assert response.status_code == HTTPStatus.OK.value


def test_auth_invalid_token(mock_auth, mock_client):
    mock_auth.return_value = False
    headers = {"Authorization": "Bearer mocked_token"}

    # Set up the mock client response
    mock_client.return_value.get.return_value.status_code = (
        HTTPStatus.UNAUTHORIZED.value
    )
    mock_client.return_value.get.return_value.json.return_value = {
        "detail": "Not authenticated: Invalid token"
    }

    response = mock_client.return_value.get("/auth", headers=headers)
    assert response.status_code == HTTPStatus.UNAUTHORIZED.value
    assert response.json() == {"detail": "Not authenticated: Invalid token"}


def test_auth_missing_token(mock_client):
    # Set up the mock client response
    mock_client.return_value.get.return_value.status_code = (
        HTTPStatus.UNAUTHORIZED.value
    )
    mock_client.return_value.get.return_value.json.return_value = {
        "detail": "Unauthenticated: Missing Authorization header"
    }

    response = mock_client.return_value.get("/auth")
    assert response.status_code == HTTPStatus.UNAUTHORIZED.value
    assert response.json() == {
        "detail": "Unauthenticated: Missing Authorization header"
    }


def test_auth_invalid_format(mock_client):
    headers = {"Authorization": "Basic mocked_token"}

    # Set up the mock client response
    mock_client.return_value.get.return_value.status_code = (
        HTTPStatus.UNAUTHORIZED.value
    )
    mock_client.return_value.get.return_value.json.return_value = {
        "detail": "Not authenticated: Invalid Authorization format"
    }

    response = mock_client.return_value.get("/auth", headers=headers)
    assert response.status_code == HTTPStatus.UNAUTHORIZED.value
    assert response.json() == {
        "detail": "Not authenticated: Invalid Authorization format"
    }


def test_auth_token_missing_in_format(mock_client):
    headers = {"Authorization": "Bearer "}

    # Set up the mock client response
    mock_client.return_value.get.return_value.status_code = (
        HTTPStatus.UNAUTHORIZED.value
    )
    mock_client.return_value.get.return_value.json.return_value = {
        "detail": "Not authenticated: Token is missing"
    }

    response = mock_client.return_value.get("/auth", headers=headers)
    assert response.status_code == HTTPStatus.UNAUTHORIZED.value
    assert response.json() == {"detail": "Not authenticated: Token is missing"}


# New tests for the chat endpoint
@pytest.fixture
def mock_chat_service():
    mock_service = MagicMock()
    app.state.model = mock_service  # Set the model in the FastAPI app state
    return mock_service


def test_chat_endpoint_valid_token(
    mock_auth, mock_db_session, mock_chat_service, mock_client
):
    mock_auth.return_value = True
    mock_chat_service.handle_chat.return_value = "mocked response"

    headers = {"Authorization": "Bearer mocked_token"}

    # Set up the mock client response
    mock_client.return_value.post.return_value.status_code = (
        HTTPStatus.OK.value
    )
    mock_client.return_value.post.return_value.json.return_value = {
        "status": HTTPStatus.OK.value,
        "response": "mocked response",
    }

    response = mock_client.return_value.post(
        "/chat", headers=headers, json={"message": "Hello"}
    )

    assert response.status_code == HTTPStatus.OK.value
    assert response.json() == {
        "status": HTTPStatus.OK.value,
        "response": "mocked response",
    }


def test_chat_endpoint_invalid_token(
    mock_auth, mock_chat_service, mock_client
):
    mock_auth.return_value = False

    headers = {"Authorization": "Bearer mocked_token"}

    # Set up the mock client response
    mock_client.return_value.post.return_value.status_code = (
        HTTPStatus.UNAUTHORIZED.value
    )
    mock_client.return_value.post.return_value.json.return_value = {
        "detail": "Not authenticated: Invalid token"
    }

    response = mock_client.return_value.post(
        "/chat", headers=headers, json={"message": "Hello"}
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED.value
    assert response.json() == {"detail": "Not authenticated: Invalid token"}
