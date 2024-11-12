import asyncio
import logging

import redis.asyncio as redis
import json
from app.config import REDIS_HOST, REDIS_PORT


class RedisClient:
    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=0):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
        )

    async def set(self, key: str, value: dict, ttl: int = 3600):
        """Set a value in Redis with an expiration time."""
        await self.client.setex(key, ttl, json.dumps(value))

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

    async def ping(self):
        """Check if the Redis server is available and responsive."""
        try:
            response = await self.client.ping()
            if response:
                logging.info("Successfully connected to Redis!")
                return True
            else:
                logging.error("Failed to connect to Redis.")
                return False
        except ConnectionError as e:
            logging.error(f"Redis connection error: {e}")
            return False


# Entry point to run the Redis connection check
async def main():
    redis_client = RedisClient()
    try:
        is_connected = await redis_client.ping()
        if is_connected:
            print("Redis is available.")
        else:
            print("Failed to connect to Redis.")
    finally:
        await redis_client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())  # This will work if no event loop is running
    except RuntimeError as error:
        if 'This event loop is already running' in str(error):
            # If an event loop is already running, use it instead
            loop = asyncio.get_event_loop()
            loop.create_task(main())
