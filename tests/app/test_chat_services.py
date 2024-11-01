import pytest
from unittest.mock import MagicMock, AsyncMock
from app.services import ChatService
from app.types import GptResponse
from app.types.enum import HTTPStatus


@pytest.fixture
def chat_service():
    db = MagicMock()
    model = AsyncMock()
    service = ChatService(db, model)
    return service


@pytest.mark.asyncio
async def test_handle_chat_missing_user_id(chat_service):
    # Setup
    request = MagicMock()
    request.json = AsyncMock(return_value={"user_id": None, "prompt": "Hello"})

    # Act
    response = await chat_service.handle_chat(request)

    # Assert
    assert response['status'] == HTTPStatus.NOT_FOUND.value
    assert response['response'] == "Invalid User ID provided."


@pytest.mark.asyncio
async def test_handle_chat_missing_prompt(chat_service):
    # Setup
    request = MagicMock()
    request.json = AsyncMock(return_value={"user_id": 1})

    # Act
    response = await chat_service.handle_chat(request)

    # Assert
    assert response['status'] == HTTPStatus.NOT_FOUND.value
    assert response['response'] == "Prompt must be a non-empty string."


@pytest.mark.asyncio
async def test_handle_chat_conversation_creation(chat_service):
    # Setup
    mock_user = MagicMock(id=1)  # Ensure correct mock setup
    chat_service.db_manager.create_user_if_not_exists = AsyncMock(
        return_value=mock_user
    )
    chat_service.db_manager.create_conversation = AsyncMock(
        return_value=MagicMock(id=1)
    )
    chat_service.db_manager.create_message_with_vector = AsyncMock()
    chat_service.db_manager.get_conversation_vector_history = AsyncMock(
        return_value=[]
    )
    chat_service.model.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    chat_service.model.get_chat_response = AsyncMock(
        return_value=GptResponse(content="Hello, how can I help you?")
    )

    request = MagicMock()
    request.json = AsyncMock(
        return_value={"user_id": 1, "prompt": "Hello", "conversation_id": None}
    )

    # Act
    response = await chat_service.handle_chat(request)

    # Assert
    chat_service.db_manager.create_conversation.assert_called_once_with(
        user_id=1
    )
    assert response['status'] == HTTPStatus.OK.value
    assert response['response'] == "Hello, how can I help you?"  # Check the actual response


@pytest.mark.asyncio
async def test_handle_chat_internal_error(chat_service):
    # Setup
    chat_service.db_manager.create_user_if_not_exists = AsyncMock(
        side_effect=Exception("Database error")
    )
    request = MagicMock()
    request.json = AsyncMock(return_value={"user_id": 1, "prompt": "Hello"})

    # Act
    response = await chat_service.handle_chat(request)

    # Assert
    assert response['status'] == HTTPStatus.INTERNAL_SERVER_ERROR.value
    assert "An error occurred: Database error" in response['response']
