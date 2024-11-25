from datetime import datetime

from app.model import User
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
        # Convert the SQLAlchemy model to a dictionary, excluding internal attributes
        user_dict = {
            key: value
            for key, value in user.__dict__.items()
            if not key.startswith('_')
        }

        # Convert datetime objects to ISO 8601 string format
        for key, value in user_dict.items():
            if isinstance(value, datetime):
                user_dict[key] = value.isoformat()

        # Now you can cache the user in Redis
        await self.redis_client.set(f"user:{user.uuid}", user_dict, ttl=3600)

    async def cache_conversation(self, user_id: int, conversation):
        if conversation:
            data = {
                "id": conversation.id,
                "conversation_order": conversation.conversation_order,
                "created_at": conversation.created_at.isoformat(),
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
