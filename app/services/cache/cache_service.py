from datetime import datetime, timezone
from typing import List
from app.model import Message, User
from app.schemas import GptResponseSchema
from app.utils import DataNormalizer


class CacheService:
    def __init__(self, redis_client):
        self.redis_client = redis_client

    async def get_user(self, user_uuid: str):
        return await self._get_from_cache(
            f"user:{user_uuid}", DataNormalizer.normalize_user
        )

    async def get_conversation(self, user_id: int, conversation_order: int):
        return await self._get_from_cache(
            f"conversation:{user_id}:{conversation_order}",
            DataNormalizer.normalize_conversation,
        )

    async def get_user_conversation(self, user_id: int, conversation_id: int):
        return await self._get_from_cache(
            f"user_conversation:{user_id}:{conversation_id}",
            DataNormalizer.normalize_user_conversation,
        )

    async def cache_user(self, user: User):
        """Caches the user in Redis."""
        user_dict = {
            key: value
            for key, value in user.__dict__.items()
            if not key.startswith('_')
        }
        for key, value in user_dict.items():
            if isinstance(value, datetime):
                user_dict[key] = value.isoformat()

        await self.redis_client.set(f"user:{user.uuid}", user_dict, ttl=3600)

    async def cache_conversation(self, user_id: int, conversation):
        if conversation:
            data = {
                "id": conversation.id,
                "conversation_order": conversation.conversation_order,
                "created_at": conversation.created_at,
                "end_at": (
                    conversation.end_at.isoformat()
                    if conversation.end_at
                    else None
                ),
            }
            await self.redis_client.set(
                f"conversation:{user_id}:{conversation.conversation_order}",
                data,
                ttl=3600,
            )

    async def cache_user_conversation(
        self, user_id: int, conversation_id: int, user_conversation
    ):
        if user_conversation:
            data = {
                "id": user_conversation.id,
                "user_id": user_conversation.user_id,
                "conversation_id": user_conversation.conversation_id,
            }
            await self.redis_client.set(
                f"user_conversation:{user_id}:{conversation_id}",
                data,
                ttl=3600,
            )

    async def _get_from_cache(self, key: str, normalizer: callable):
        data = await self.redis_client.get(key)
        return normalizer(data) if data else None

    async def cache_conversation_history(
        self, user_conversation_id: int, conversation_history: List[Message]
    ):
        """Cache the conversation history in Redis."""
        key = f"conversation_history:{user_conversation_id}"
        # Manually create a list of serialized Message objects
        history_data = [
            {
                "id": msg.id,
                "content": msg.content,
                "created_at": msg.created_at,
                "user_conversation_id": msg.user_conversation_id,
            }
            for msg in conversation_history
        ]
        await self.redis_client.set(key, history_data, ttl=3600)

    async def get_conversation_history(self, user_conversation_id: int):
        """Retrieve conversation history from cache."""
        key = f"conversation_history:{user_conversation_id}"
        history_data = await self.redis_client.get(key)
        if history_data:
            return [Message(**msg) for msg in history_data]
        return None

    async def cache_message(self, user_conversation_id: int, message):
        """Cache a new message to the conversation history."""
        key = f"conversation_history:{user_conversation_id}"

        # Retrieve existing history
        history_data = await self.redis_client.get(key) or []

        # Check if the message is a database Message object or GptResponseSchema
        if isinstance(message, Message):
            # Prepare new message data from a Message object
            new_message = {
                "id": message.id,
                "content": message.content,
                "created_at": message.created_at,
                "user_conversation_id": message.user_conversation_id,
            }
        elif isinstance(message, GptResponseSchema):
            # Prepare new message data from GptResponseSchema
            new_message = {
                "role": None,
                "content": message.content,
                "created_at": datetime.now(timezone.utc),
                "user_conversation_id": user_conversation_id,
            }
        else:
            raise ValueError("Invalid message type for caching.")

        # Append the new message to the existing history
        history_data.append(new_message)

        # Cache the updated history
        await self.redis_client.set(key, history_data, ttl=3600)
