import json
import time

from app.database.database_manager import DatabaseManager
from app.services.cache import CacheService

from app.schemas import UserPromptSchema
from app.client import RedisClient
from app.llm import BlueViAgent
from fastapi import Request
import logging

from app.types.enum.gpt import MessageType, Role
from app.types.enum.http_status import HTTPStatus
from app.utils import ResponseUtils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, db, user_prompt: UserPromptSchema, request: Request):
        self.db_manager = DatabaseManager(db)
        self.cache_service = CacheService(RedisClient())
        self.response_utils = ResponseUtils()
        self.user = user_prompt
        self.agent = BlueViAgent(
            model=request.app.state.model,
            db_manager=self.db_manager,
            cache_service=self.cache_service,
        )

    async def handle_chat(self) -> dict:
        try:
            start_time = time.time()
            user = await self._get_or_create_user()
            conversation = await self._get_or_create_conversation(user.id)
            user_conversation = await self._get_or_create_user_conversation(
                user.id, conversation.id
            )

            # Cache the user's prompt
            user_message = self.db_manager.create_message(
                user_conversation.id,
                self.user.prompt,
                MessageType.PROMPT,
                Role.USER,
            )
            if not user_message:
                return self.response_utils.error_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "Failed to store user message.",
                )
            await self.cache_service.cache_message(
                user_conversation.id, user_message
            )

            # Generate and cache bot response
            bot_response = await self.agent.handle_conversation(user_message)
            await self.cache_service.cache_message(
                user_conversation.id, bot_response
            )
            # Calculate the time taken
            end_time = time.time()
            time_taken = end_time - start_time

            # Safely append "response_time" to dynamic_json
            if hasattr(bot_response, "dynamic_json"):
                if bot_response.dynamic_json is None:
                    # If dynamic_json is None, initialize it as a dictionary with response_time
                    bot_response.dynamic_json = {"response_time": time_taken}
                elif isinstance(bot_response.dynamic_json, dict):
                    # If it's already a dictionary, add the new field
                    bot_response.dynamic_json["response_time"] = time_taken
                else:
                    try:
                        # Try parsing it as JSON if it's a string
                        dynamic_data = bot_response.dynamic_json
                        dynamic_data["response_time"] = time_taken
                        bot_response.dynamic_json = dynamic_data
                    except (TypeError, json.JSONDecodeError):
                        # If parsing fails, replace it with a new dictionary
                        bot_response.dynamic_json = {
                            "response_time": time_taken
                        }
            else:
                # If no dynamic_json attribute, create one
                bot_response.dynamic_json = {"response_time": time_taken}

            return self.response_utils.success_response(
                bot_response, conversation.conversation_order
            )
        except Exception as e:
            logger.error(f"Error in handle_chat: {str(e)}", exc_info=True)
            return self.response_utils.error_response(
                HTTPStatus.INTERNAL_SERVER_ERROR, "Request processing error."
            )

    async def _get_or_create_user(self):
        """Handles user retrieval or creation."""
        cached_user = await self.cache_service.get_user(self.user.uuid)
        if cached_user:
            return cached_user
        user = self.db_manager.get_user(self.user.uuid)
        if not user:
            raise ValueError("Invalid user ID.")
        await self.cache_service.cache_user(user)
        return user

    async def _get_or_create_conversation(self, user_id: int):
        """Handles conversation retrieval or creation."""
        cached_conversation = await self.cache_service.get_conversation(
            user_id, self.user.conversation_order
        )
        if cached_conversation:
            return cached_conversation
        conversation = self.db_manager.get_or_create_conversation(
            user_id, self.user.conversation_order
        )
        await self.cache_service.cache_conversation(user_id, conversation)
        return conversation

    async def _get_or_create_user_conversation(
        self, user_id: int, conversation_id: int
    ):
        """Handles user-conversation relationship retrieval or creation."""
        cached_user_conversation = (
            await self.cache_service.get_user_conversation(
                user_id, conversation_id
            )
        )
        if cached_user_conversation:
            return cached_user_conversation
        user_conversation = self.db_manager.get_user_conversation(
            user_id, conversation_id
        )
        if not user_conversation:
            user_conversation = self.db_manager.create_user_conversation(
                user_id, conversation_id
            )
        await self.cache_service.cache_user_conversation(
            user_id, conversation_id, user_conversation
        )
        return user_conversation
