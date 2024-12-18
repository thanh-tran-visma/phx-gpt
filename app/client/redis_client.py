from datetime import datetime

import redis.asyncio as redis
import json
from app.config import REDIS_HOST, REDIS_PORT


class RedisClient:
    def __init__(self, db=0):
        self.client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=db,
            decode_responses=True,
        )

    async def set(self, key: str, value: dict, ttl: int = 3600):
        """Set a value in Redis with an expiration time."""

        # Function to convert datetime objects to ISO 8601 strings
        def datetime_converter(o):
            if isinstance(o, datetime):
                return o.isoformat()

        # Serialize the value using json.dumps, with custom handling for datetime objects
        await self.client.setex(
            key, ttl, json.dumps(value, default=datetime_converter)
        )

    async def get(self, key: str):
        """Get a value from Redis."""
        data = await self.client.get(key)
        return json.loads(data) if data else None

    async def delete(self, key: str):
        """Delete a value from Redis."""
        await self.client.delete(key)

    async def close(self):
        """Close the Redis connection asynchronously."""
        await self.client.aclose()
