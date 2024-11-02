import pytest
from unittest.mock import MagicMock
from app.database.database_manager import DatabaseManager
from app.model import User, Conversation, Message


@pytest.fixture
def mock_db_session(mocker):
    """Fixture to create a mock database session."""
    return mocker.MagicMock()


@pytest.fixture
def database_manager(mock_db_session):
    """Fixture to create a DatabaseManager instance with a mocked DB session."""
    return DatabaseManager(db=mock_db_session)


def test_user_exists(database_manager, mock_db_session):
    user_id = 1
    mock_db_session.query.return_value.filter.return_value.count.return_value = (
        1
    )
    assert database_manager.user_exists(user_id) is True
    mock_db_session.query.return_value.filter.return_value.count.return_value = (
        0
    )
    assert database_manager.user_exists(user_id) is False


def test_create_user_if_not_exists(database_manager, mock_db_session):
    user_id = 1
    user_mock = MagicMock(spec=User)
    database_manager.user_manager.get_user = MagicMock(return_value=None)
    database_manager.user_manager.create_user_if_not_exists = MagicMock(
        return_value=user_mock
    )
    created_user = database_manager.create_user_if_not_exists(user_id)
    database_manager.user_manager.get_user.assert_called_once_with(user_id)
    database_manager.user_manager.create_user_if_not_exists.assert_called_once_with(
        user_id
    )
    assert created_user == user_mock


def test_create_conversation(database_manager, mock_db_session):
    user_id = 1
    conversation_mock = MagicMock(spec=Conversation)
    database_manager.conversation_manager.create_conversation = MagicMock(
        return_value=conversation_mock
    )
    created_conversation = database_manager.create_conversation(user_id)
    database_manager.conversation_manager.create_conversation.assert_called_once_with(
        user_id
    )
    assert created_conversation == conversation_mock


def test_delete_conversation(database_manager, mock_db_session):
    conversation_id = 1
    database_manager.conversation_manager.delete_conversation = MagicMock()
    database_manager.delete_conversation(conversation_id)
    database_manager.conversation_manager.delete_conversation.assert_called_once_with(
        conversation_id
    )


def test_create_message_with_vector(database_manager, mock_db_session):
    conversation_id = 1
    content = "Hello!"
    message_type = "text"
    embedding_vector = [0.1, 0.2, 0.3]
    role = "user"
    message_mock = MagicMock(spec=Message)
    database_manager.message_manager.create_message_with_vector = MagicMock(
        return_value=message_mock
    )
    created_message = database_manager.create_message_with_vector(
        conversation_id, content, message_type, embedding_vector, role
    )
    database_manager.message_manager.create_message_with_vector.assert_called_once_with(
        conversation_id, content, message_type, embedding_vector, role
    )
    assert created_message == message_mock


def test_get_conversation_vector_history(database_manager, mock_db_session):
    conversation_id = 1
    messages_mock = [MagicMock(spec=Message)]
    database_manager.message_manager.get_messages_by_conversation = MagicMock(
        return_value=messages_mock
    )
    retrieved_messages = database_manager.get_conversation_vector_history(
        conversation_id
    )
    database_manager.message_manager.get_messages_by_conversation.assert_called_once_with(
        conversation_id
    )
    assert retrieved_messages == messages_mock
