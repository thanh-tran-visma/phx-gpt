import logging
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_huggingface import HuggingFaceEndpoint
from app.config import GPT_ENDPOINT_URL
from app.llm.blue_vi_assistant import BlueViGptAssistant
from app.config.config_env import (
    HF_TOKEN,
    LLM_MAX_TOKEN,
)
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
        """Load the HuggingFace endpoint model with debug information."""
        try:
            callbacks = [StreamingStdOutCallbackHandler()]
            logging.info("Connecting to Hugging Face endpoint.")
            model = HuggingFaceEndpoint(
                huggingfacehub_api_token=HF_TOKEN,
                max_new_tokens=LLM_MAX_TOKEN,
                callbacks=callbacks,
                streaming=False,
                stop=["<|eot_id|>"],
                endpoint_url=GPT_ENDPOINT_URL,
            )
            logging.info("Model connected successfully. Model details:")
            logging.info(model)
            return model
        except Exception as e:
            logging.error(f"Error connecting to Hugging Face: {e}")
            raise RuntimeError(
                "Failed to connect to HuggingFaceEndpoint."
            ) from e

    def close(self):
        """Close and clean up resources."""
        if hasattr(self.llm, "close"):
            try:
                self.llm.close()
                logging.info("LLM resources closed successfully.")
            except Exception as e:
                logging.error(f"Error during LLM resource cleanup: {e}")
