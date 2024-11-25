import os
from dotenv import load_dotenv

# Load environment variables only once
load_dotenv()


class EnvConfig:
    @staticmethod
    def get(key: str, default=None):
        """
        Retrieve the value of an environment variable.

        :param key: The environment variable name.
        :param default: The default value if the environment variable is not found.
        :return: The value of the environment variable, or default if not found.
        """
        return os.getenv(key, default)

    @staticmethod
    def get_int(key: str, default=None):
        """
        Retrieve the value of an environment variable and convert it to an integer.

        :param key: The environment variable name.
        :param default: The default value if the environment variable is not found.
        :return: The integer value of the environment variable, or default if not found.
        """
        return (
            int(os.getenv(key, default))
            if os.getenv(key, default) is not None
            else None
        )


HF_TOKEN = EnvConfig.get("HF_TOKEN")
MODEL_NAME = EnvConfig.get("MODEL_NAME")
GGUF_MODEL = EnvConfig.get("GGUF_MODEL")
BLUEVI_GPT = EnvConfig.get("BLUEVI_GPT")
BEARER_TOKEN = EnvConfig.get("BEARER_TOKEN")
API_URL = EnvConfig.get("API_URL")

LLM_MAX_TOKEN = EnvConfig.get_int("LLM_MAX_TOKEN", 2048)
MAX_HISTORY_WINDOW_SIZE = EnvConfig.get_int("MAX_HISTORY_WINDOW_SIZE", 512)

REDIS_HOST = EnvConfig.get("REDIS_HOST")
REDIS_PORT = EnvConfig.get_int("REDIS_PORT")

DB_HOST = EnvConfig.get("DB_HOST", "localhost")
DB_PORT = EnvConfig.get_int("DB_PORT", 3306)
DB_DATABASE = EnvConfig.get("DB_DATABASE")
DB_USERNAME = EnvConfig.get("DB_USERNAME")
DB_PASSWORD = EnvConfig.get("DB_PASSWORD", "")
