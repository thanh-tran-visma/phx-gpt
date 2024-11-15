import logging

from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.database import DatabaseManager
from app.llm import BlueViGptModel
from app.types.enum.http_status import HTTPStatus

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EndConversationService:
    def __init__(self, db: Session):
        self.db = db
        self.db_manager = DatabaseManager(db)
        self.blue_vi_gpt_model = BlueViGptModel()

    async def handle_end_conversation(
        self, uuid: str, conversation_order: int
    ):
        logger.debug(
            f"Handling end conversation for user {uuid} with conversation order {conversation_order}"
        )

        user = self.db_manager.get_user(uuid)
        if not user:
            logger.error(f"User with UUID {uuid} not found")
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="User not found",
            )
        logger.debug(f"User found: {user.id}")

        conversation = (
            self.db_manager.get_conversation_by_user_id_and_conversation_order(
                user.id, conversation_order
            )
        )
        if not conversation:
            logger.error(
                f"Conversation with order {conversation_order} for user {user.id} not found"
            )
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="Conversation not found",
            )
        logger.debug(
            f"Conversation found: {conversation.id}, order {conversation.conversation_order}"
        )

        user_conversation = self.db_manager.get_user_conversation(
            user.id, conversation.id
        )
        if not user_conversation:
            logger.error(
                f"UserConversation for user {user.id} and conversation {conversation.id} not found"
            )
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="UserConversation not found",
            )
        logger.debug(f"UserConversation found: {user_conversation.id}")

        # Handle sensitive messages
        await self.handle_sensitive_data(user_conversation.id)

        # End the conversation and return a success response if completed
        logger.debug(f"Ending conversation {conversation.id}")
        return self.db_manager.end_conversation(conversation.id)

    async def handle_sensitive_data(self, user_conversation_id: int):
        logger.debug(
            f"Handling sensitive data for user conversation {user_conversation_id}"
        )
        sensitive_messages = self.db_manager.get_sensitive_messages(
            user_conversation_id
        )
        logger.debug(f"Found {len(sensitive_messages)} sensitive messages")

        # Process each sensitive message for anonymization
        for message in sensitive_messages:
            logger.debug(f"Checking message {message.id} for personal data")
            logger.debug(
                f"Sensitive data detected in message {message.id}, anonymizing"
            )
            anonymized_message = await self.blue_vi_gpt_model.assistant_role.get_anonymized_message(
                message.content
            )
            self.db_manager.update_message_content(
                message.id, anonymized_message.content
            )
            logger.debug(f"Anonymized message {message.id}")
