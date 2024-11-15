import pytest
from unittest.mock import create_autospec
from sqlalchemy.orm import Session

from app.database import DatabaseManager
from app.model import User, Message, Conversation, UserConversation
from app.types.enum import Role


@pytest.fixture
def mock_db_session():
    return create_autospec(Session)


@pytest.fixture
def db_manager(mock_db_session):
    return DatabaseManager(mock_db_session)


def test_get_user(db_manager, mock_db_session):
    user_id = 1
    uuid = 'cfb6e466-8366-4f88-bdf9-3ae6984c0716'
    mock_db_session.query.return_value.filter.return_value.first.return_value = User(
        id=user_id, uuid=uuid
    )
    test_user = db_manager.get_user(uuid)
    assert test_user.id == user_id
    assert test_user.uuid == uuid
    mock_db_session.query.assert_called_once_with(User)


def test_create_user_if_not_exists_creates_user(db_manager, mock_db_session):
    uuid = 'cfb6e466-8366-4f88-bdf9-3ae6984c0716'
    mock_db_session.query.return_value.filter.return_value.first.return_value = (
        None
    )

    new_user = db_manager.create_user_if_not_exists(uuid)

    assert new_user is not None
    assert new_user.uuid == uuid
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()


def test_get_conversation_by_order(db_manager, mock_db_session):
    user_id = 1
    conversation_order = 1
    mock_conversation = Conversation(
        user_id=user_id, conversation_order=conversation_order
    )
    mock_db_session.query.return_value.filter.return_value.first.return_value = (
        mock_conversation
    )

    conversation = (
        db_manager.get_conversation_by_user_id_and_conversation_order(
            user_id, conversation_order
        )
    )

    assert conversation == mock_conversation


def test_create_conversation(db_manager, mock_db_session):
    user_id = 1
    mock_db_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = (
        None  # No existing conversations
    )

    new_conversation = db_manager.create_conversation(user_id)

    assert new_conversation is not None
    assert new_conversation.user_id == user_id
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()


def test_create_message(db_manager, mock_db_session):
    user_conversation_id = 1
    content = "Hello"
    message_type = "text"
    role = Role.USER.value

    Message(content=content, message_type=message_type, role=role)
    mock_db_session.add.return_value = None

    new_message = db_manager.create_message(
        user_conversation_id, content, message_type, role
    )

    assert new_message is not None
    assert new_message.content == content
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()


def test_get_messages_by_user_conversation_id(db_manager, mock_db_session):
    user_conversation_id = 1
    mock_messages = [Message(content="Hello"), Message(content="World")]
    mock_db_session.query.return_value.filter.return_value.all.return_value = (
        mock_messages
    )

    messages = db_manager.get_messages_by_user_conversation_id(
        user_conversation_id
    )

    assert messages == mock_messages


def test_get_conversations_for_user(db_manager, mock_db_session):
    user_id = 1
    mock_conversations = [UserConversation(user_id=user_id, conversation_id=1)]
    mock_db_session.query.return_value.filter.return_value.all.return_value = (
        mock_conversations
    )

    conversations = db_manager.get_conversations_for_user(user_id)

    assert conversations == mock_conversations


def test_create_user_conversation(db_manager, mock_db_session):
    user_id = 1
    conversation_id = 1

    user_conversation = db_manager.create_user_conversation(
        user_id, conversation_id
    )

    assert user_conversation is not None
    assert user_conversation.user_id == user_id
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()


def test_check_user_conversation_exists(db_manager, mock_db_session):
    user_id = 1
    conversation_id = 1
    mock_db_session.query.return_value.filter.return_value.count.return_value = (
        1
    )

    exists = db_manager.check_user_conversation_exists(
        user_id, conversation_id
    )

    assert exists is True
