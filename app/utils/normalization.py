from app.model import User, Conversation, UserConversation


class DataNormalizer:
    @staticmethod
    def normalize_user(data: dict):
        return User(**data) if data else None

    @staticmethod
    def normalize_conversation(data: dict):
        return Conversation(**data) if data else None

    @staticmethod
    def normalize_user_conversation(data: dict):
        return UserConversation(**data) if data else None
