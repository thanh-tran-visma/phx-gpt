import redis.asyncio as redis
import json
import logging
from app.config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD


class RedisClient:
    def __init__(
        self, host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db=0
    ):
        # Initialize the Redis client with the given parameters
        self.client = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=True,
        )

    async def ping(self):
        """Check if the Redis server is available and responsive."""
        try:
            # Perform a ping to Redis
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


# Entry point to run the Redis connection check
async def main():
    # Create the Redis client instance
    redis_client = RedisClient()

    # Check the connection to Redis
    is_connected = await redis_client.ping()

    if is_connected:
        print("Redis is available.")
    else:
        print("Failed to connect to Redis.")


# If this script is being run directly, execute the main function
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
