from typing import Optional
from sqlalchemy.orm import Session
from app.client import RedisClient
from app.database import DatabaseManager
from app.llm import BlueViAgent
from app.model import Conversation, User, UserConversation
from app.schemas import UserPromptSchema, GptResponseSchema
from app.types.enum.http_status import HTTPStatus
from app.types.enum.gpt import MessageType, Role
from fastapi import Request
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self, db: Session, user_prompt: UserPromptSchema, request: Request
    ):
        self.db_manager = DatabaseManager(db)
        self.user = user_prompt
        self.agent = BlueViAgent(
            model=request.app.state.model,
            db_manager=self.db_manager,
        )
        self.redis_client = RedisClient()

    async def handle_chat(self) -> dict:
        """Handles the chat request, managing user, conversation, and message processing."""
        try:
            user = await self._get_or_create_user()
            conversation = await self._get_or_create_conversation(user.id)
            user_conversation = await self._get_or_create_user_conversation(
                user.id, conversation.id
            )

            message = self._create_user_message(user_conversation.id)
            if not message:
                return self._error_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "Failed to store user message.",
                )

            bot_response = await self.agent.handle_conversation(message)
            self._store_bot_response(user_conversation.id, bot_response)

            return self._success_response(
                bot_response, conversation.conversation_order
            )

        except Exception as e:
            logger.error(f"Error in handle_chat: {str(e)}", exc_info=True)
            return self._error_response(
                HTTPStatus.INTERNAL_SERVER_ERROR, "Request processing error."
            )

    async def _get_or_create_user(self) -> User:
        """Retrieves the user from cache or database, creating if necessary."""
        cached_user = await self._get_from_cache(
            f"user:{self.user.uuid}", self._normalize_user_cache
        )
        if cached_user:
            logger.info(f"User found in cache: {cached_user}")
            return cached_user

        user = self.db_manager.get_user(self.user.uuid)
        if not user:
            raise ValueError("Invalid user ID.")
        await self._cache_user(user)
        return user

    async def _get_or_create_conversation(self, user_id: int) -> Conversation:
        """Retrieves or creates a conversation."""
        cache_key = f"conversation:{user_id}:{self.user.conversation_order}"
        cached_conversation = await self._get_from_cache(
            cache_key, self._normalize_conversation_cache
        )
        if cached_conversation:
            logger.info(f"Conversation found in cache: {cached_conversation}")
            return cached_conversation

        conversation = self.db_manager.get_or_create_conversation(
            user_id, self.user.conversation_order
        )
        if not conversation:
            raise ValueError("Failed to create or retrieve the conversation.")
        await self._cache_conversation(user_id, conversation)
        return conversation

    async def _get_or_create_user_conversation(
        self, user_id: int, conversation_id: int
    ) -> UserConversation:
        """Retrieves or creates a user conversation."""
        cache_key = f"user_conversation:{user_id}:{conversation_id}"
        cached_user_conversation = await self._get_from_cache(
            cache_key, self._normalize_user_conversation_cache
        )
        if cached_user_conversation:
            logger.info(
                f"User conversation found in cache: {cached_user_conversation}"
            )
            return cached_user_conversation

        user_conversation = self.db_manager.get_user_conversation(
            user_id, conversation_id
        )
        if not user_conversation:
            user_conversation = self.db_manager.create_user_conversation(
                user_id, conversation_id
            )
        await self._cache_user_conversation(
            user_id, conversation_id, user_conversation
        )
        return user_conversation

    def _create_user_message(self, user_conversation_id: int):
        """Creates a user message."""
        return self.db_manager.create_message(
            user_conversation_id,
            self.user.prompt,
            MessageType.PROMPT,
            Role.USER,
        )

    def _store_bot_response(
        self, user_conversation_id: int, bot_response: GptResponseSchema
    ):
        """Stores the bot's response."""
        self.db_manager.create_message(
            user_conversation_id,
            bot_response.content,
            MessageType.RESPONSE,
            Role.ASSISTANT,
        )

    async def _get_from_cache(self, key: str, normalizer: callable):
        """Generic method to fetch and normalize data from Redis cache."""
        cached_data = await self.redis_client.get(key)
        return await normalizer(cached_data) if cached_data else None

    @staticmethod
    async def _normalize_user_cache(data: dict) -> Optional[User]:
        return User(**data) if data else None

    @staticmethod
    async def _normalize_conversation_cache(
        data: dict,
    ) -> Optional[Conversation]:
        return Conversation(**data) if data else None

    @staticmethod
    async def _normalize_user_conversation_cache(
        data: dict,
    ) -> Optional[UserConversation]:
        return UserConversation(**data) if data else None

    async def _cache_user(self, user: User):
        """Caches the user in Redis."""
        await self.redis_client.set(f"user:{user.uuid}", user.dict(), ttl=3600)

    async def _cache_conversation(self, user_id: int, conversation: Optional[Conversation]):
        """Caches the conversation in Redis."""
        if conversation:
            conversation_data = {
                "id": conversation.id,
                "conversation_order": conversation.conversation_order,
                "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
                "end_at": conversation.end_at.isoformat() if conversation.end_at else None,
            }
            await self.redis_client.set(
                f"conversation:{user_id}:{conversation.conversation_order}",
                conversation_data,
                ttl=3600,
            )

    async def _cache_user_conversation(
            self,
            user_id: int,
            conversation_id: int,
            user_conversation: Optional[UserConversation],
    ):
        """Caches the user conversation in Redis."""
        if user_conversation:
            user_conversation_data = {
                "id": user_conversation.id,
                "user_id": user_conversation.user_id,
                "conversation_id": user_conversation.conversation_id,
            }
            await self.redis_client.set(
                f"user_conversation:{user_id}:{conversation_id}",
                user_conversation_data,
                ttl=3600,
            )

    @staticmethod
    def _success_response(
        bot_response: GptResponseSchema, conversation_order: int
    ) -> dict:
        return {
            "status": bot_response.status,
            "response": bot_response.content,
            "conversation_order": conversation_order,
            "dynamic_json": bot_response.dynamic_json,
            "type": bot_response.type,
        }

    @staticmethod
    def _error_response(status: HTTPStatus, message: str) -> dict:
        return {"status": status.value, "response": message}
