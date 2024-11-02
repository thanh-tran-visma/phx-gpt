import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from starlette.requests import Request
from http import HTTPStatus
from app.database import DatabaseManager
from app.services import ChatService
from app.types import GptResponse


@pytest.fixture
def mock_user_manager():
    return MagicMock()


@pytest.fixture
def mock_message_manager():
    return MagicMock()


@pytest.fixture
def mock_conversation_manager():
    return MagicMock()


@pytest.fixture
def mock_db_session():
    return MagicMock()


@pytest.fixture
def mock_database_manager(
    mock_user_manager,
    mock_message_manager,
    mock_conversation_manager,
    mock_db_session,
):
    with patch(
        'app.database.model_managers.UserManager',
        return_value=mock_user_manager,
    ), patch(
        'app.database.model_managers.MessageManager',
        return_value=mock_message_manager,
    ), patch(
        'app.database.model_managers.ConversationManager',
        return_value=mock_conversation_manager,
    ):
        db_manager = DatabaseManager(mock_db_session)
        db_manager.user_manager = mock_user_manager
        db_manager.message_manager = mock_message_manager
        db_manager.conversation_manager = mock_conversation_manager
        yield db_manager


@pytest.fixture
def mock_model():
    return MagicMock()


@pytest.fixture
def chat_service(mock_database_manager, mock_model):
    return ChatService(db=mock_database_manager.db, model=mock_model)


@pytest.mark.asyncio
async def test_handle_chat_invalid_user_id(chat_service):
    request = MagicMock(spec=Request)
    request.json = AsyncMock(
        return_value={
            "prompt": "Hello",
            "user_id": None,
            "conversation_id": None,
        }
    )

    response = await chat_service.handle_chat(request)

    assert response["status"] == HTTPStatus.NOT_FOUND.value
    assert response["response"] == "Invalid User ID provided."


@pytest.mark.asyncio
async def test_handle_chat_empty_prompt(chat_service):
    request = MagicMock(spec=Request)
    request.json = AsyncMock(
        return_value={"prompt": "", "user_id": 1, "conversation_id": None}
    )

    response = await chat_service.handle_chat(request)

    assert response["status"] == HTTPStatus.NOT_FOUND.value
    assert response["response"] == "Prompt must be a non-empty string."


@pytest.mark.asyncio
async def test_handle_chat_exception_handling(chat_service):
    request = MagicMock(spec=Request)
    request.json = AsyncMock(side_effect=Exception("Some error"))

    response = await chat_service.handle_chat(request)

    assert response["status"] == HTTPStatus.INTERNAL_SERVER_ERROR.value
    assert response["response"] == "An error occurred: Some error"


@pytest.mark.asyncio
async def test_handle_chat_valid_request(
    chat_service, mock_model, mock_database_manager
):
    # Create a mock request with JSON data
    request = MagicMock(spec=Request)
    request.json = AsyncMock(
        return_value={"prompt": "Hello", "user_id": 1, "conversation_id": None}
    )

    # Mock the user and conversation objects
    mock_user = MagicMock(id=1)
    mock_conversation = MagicMock(id=2)

    # Set up the mocks for the database manager
    mock_database_manager.user_manager.create_user_if_not_exists = AsyncMock(
        return_value=mock_user
    )
    mock_database_manager.conversation_manager.create_conversation = AsyncMock(
        return_value=mock_conversation
    )
    mock_database_manager.message_manager.create_message_with_vector = (
        AsyncMock()
    )
    mock_database_manager.get_conversation_vector_history = AsyncMock(
        return_value=[]
    )

    # Mock the model methods
    mock_model.embed.return_value = [0.1, 0.2, 0.3]
    mock_model.get_chat_response.return_value = GptResponse(
        content="Hi there!"
    )

    # Call the method under test
    response = await chat_service.handle_chat(request)

    # Assert the response status and content
    assert response["status"] == HTTPStatus.OK.value
    assert response["response"] == "Hi there!"

    # Verify that create_user_if_not_exists was called once with the correct user_id
    mock_database_manager.user_manager.create_user_if_not_exists.assert_called_once_with(
        1
    )


@pytest.mark.asyncio
async def test_handle_chat_conversation_creation_failure(
    chat_service, mock_model, mock_database_manager
):
    # Create a mock request with JSON data
    request = MagicMock(spec=Request)
    request.json = AsyncMock(
        return_value={"prompt": "Hello", "user_id": 1, "conversation_id": None}
    )

    # Mock the user object
    mock_user = MagicMock(id=1)
    # Ensure the user is created
    mock_database_manager.user_manager.create_user_if_not_exists = AsyncMock(
        return_value=mock_user
    )

    # Simulate conversation creation failure
    mock_database_manager.conversation_manager.create_conversation = AsyncMock(
        return_value=None
    )

    # Call the method under test
    response = await chat_service.handle_chat(request)

    assert response["status"] == HTTPStatus.NOT_FOUND.value
    assert response["response"] == "Failed to create a conversation."
