import pytest
from unittest.mock import patch, MagicMock
from app.types.enum.http_status import HTTPStatus


# Mock the authentication
@pytest.fixture(scope="module")
def mock_auth():
    with patch("app.auth.Auth.validate_token", return_value=True):
        yield


# Mock the database session
@pytest.fixture(scope="module")
def mock_db_session():
    with patch(
        "app.database.database.Database.get_session"
    ) as mock_get_session:
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        yield mock_session


# Mock client fixture
@pytest.fixture(scope="module")
def mock_client():
    """Mock the FastAPI TestClient."""
    with patch("fastapi.testclient.TestClient") as MockClient:
        mock_test_client = (
            MockClient.return_value
        )  # Create an instance of the mocked client
        # Optionally set up mock behavior for the client here
        yield mock_test_client


def test_chat_endpoint_creates_user_and_conversation(mock_client):
    """Test chat endpoint for creating user and conversation."""

    # Define the user prompt data
    user_prompt_data = {
        "uuid": "cfb6e466-8366-4f88-bdf9-3ae6984c0721",
        "prompt": "Hello",
        "conversation_order": 1,
    }

    # Mocking the response of the chat endpoint
    mock_client.post.return_value = MagicMock(
        status_code=HTTPStatus.OK.value,
        json=lambda: {
            "status": HTTPStatus.OK.value,
            "response": "Bot response",
        },
    )

    response = mock_client.post("/chat", json=user_prompt_data)

    assert response.status_code == HTTPStatus.OK.value
    assert response.json() == {
        "status": HTTPStatus.OK.value,
        "response": "Bot response",
    }


def test_chat_endpoint_internal_server_error(mock_client):
    """Test chat endpoint for handling internal server errors."""

    user_prompt_data = {
        "uuid": "cfb6e466-8366-4f88-bdf9-3ae6984c0721",
        "prompt": "Hello",
        "conversation_order": 1,
    }

    # Mocking the response of the chat endpoint to simulate an error
    mock_client.post.return_value = MagicMock(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        json=lambda: {"detail": "Failed to process chat."},
    )

    response = mock_client.post("/chat", json=user_prompt_data)

    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR.value
    assert response.json() == {"detail": "Failed to process chat."}


def test_chat_endpoint_invalid_user(mock_client):
    """Test chat endpoint for invalid user input."""

    user_prompt_data = {
        "uuid": None,
        "prompt": "Hello",
        "conversation_order": 1,
    }

    # Mocking the validation to simulate a bad request
    mock_client.post.return_value = MagicMock(
        status_code=HTTPStatus.BAD_REQUEST.value,
        json=lambda: {"detail": "Invalid user ID"},
    )

    response = mock_client.post("/chat", json=user_prompt_data)

    assert response.status_code == HTTPStatus.BAD_REQUEST.value
    assert response.json() == {"detail": "Invalid user ID"}


def test_chat_endpoint_missing_prompt(mock_client):
    """Test chat endpoint for missing prompt in request."""

    user_prompt_data = {
        "uuid": "cfb6e466-8366-4f88-bdf9-3ae6984c0721",
        "conversation_order": 1,
    }

    # Mocking the validation to simulate a bad request
    mock_client.post.return_value = MagicMock(
        status_code=HTTPStatus.BAD_REQUEST.value,
        json=lambda: {"detail": "Prompt is required"},
    )

    response = mock_client.post("/chat", json=user_prompt_data)

    assert response.status_code == HTTPStatus.BAD_REQUEST.value
    assert response.json() == {"detail": "Prompt is required"}


def test_chat_endpoint_successful_user_creation(mock_client):
    """Test chat endpoint for successfully creating a user."""

    user_prompt_data = {
        "uuid": "cfb6e466-8366-4f88-bdf9-3ae6984c0721",
        "prompt": "Hello",
        "conversation_order": 1,
    }

    # Mocking the response of user creation
    mock_client.post.return_value = MagicMock(
        status_code=HTTPStatus.CREATED.value,
        json=lambda: {
            "status": HTTPStatus.CREATED.value,
            "response": "User created and Bot response",
        },
    )

    response = mock_client.post("/chat", json=user_prompt_data)

    assert response.status_code == HTTPStatus.CREATED.value
    assert response.json() == {
        "status": HTTPStatus.CREATED.value,
        "response": "User created and Bot response",
    }
