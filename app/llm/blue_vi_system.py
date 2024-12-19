import logging
from openai import OpenAI
from app.config import GPT_ENDPOINT_URL
from app.config.config_env import HF_TOKEN, LLM_MAX_TOKEN
from app.llm.blue_vi_assistant import BlueViGptAssistant


class BlueViGptModel:
    def __init__(self):
        """Initialize BlueViGptModel with main model and embedding model."""
        try:
            self.llm = self.load_model()
            self.assistant = BlueViGptAssistant(self.llm)
            logging.info("BlueViGptModel initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize BlueViGptModel: {e}")
            raise

    @staticmethod
    def load_model():
        try:
            logging.info(
                "Connecting to Hugging Face endpoint using OpenAI client."
            )
            client = OpenAI(base_url=GPT_ENDPOINT_URL, api_key=HF_TOKEN)
            logging.info(
                "OpenAI client connected successfully. Configuring model settings."
            )

            return {
                "client": client,
                "model": "tgi",
                "max_new_tokens": LLM_MAX_TOKEN,
                "stop": ["<|eot_id|>"],
                "response_format": "json",
            }
        except Exception as e:
            logging.error(
                f"Error connecting to Hugging Face using OpenAI client: {e}"
            )
            raise RuntimeError(
                "Failed to connect to the Hugging Face endpoint."
            ) from e

    @staticmethod
    def close():
        """Close and clean up resources."""
        try:
            logging.info("Closing OpenAI client resources.")
            # No specific close method for OpenAI client, but placeholder for any resource cleanup.
        except Exception as e:
            logging.error(f"Error during OpenAI client resource cleanup: {e}")
