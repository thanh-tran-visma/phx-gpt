import logging
from llama_cpp import Llama, LlamaTokenizer
from app.config.config_env import MODEL_NAME, HF_TOKEN, GGUF_MODEL
from app.llm.blue_vi_assistant import BlueViGptAssistant
from app.llm.blue_vi_user import BlueViGptUserManager


class BlueViGptModel:
    def __init__(self):
        """Initialize BlueViGptModel with main model and embedding model."""
        self.llm = self.load_model()
        self.user = BlueViGptUserManager(self.llm)
        self.assistant = BlueViGptAssistant(self.llm)

    @staticmethod
    def load_model() -> Llama:
        """Load the main Llama model from Hugging Face."""
        model_cache_dir = "./model_cache"
        try:
            llm = Llama.from_pretrained(
                repo_id=MODEL_NAME,
                filename=GGUF_MODEL,
                cache_dir=model_cache_dir,
                token=HF_TOKEN,
            )
            return llm
        except Exception as e:
            logging.error(f"Error loading the model: {e}")
            raise

    @property
    def tokenizer(self) -> LlamaTokenizer:
        """Return the Llama tokenizer for this model."""
        return LlamaTokenizer(self.llm)

    def close(self):
        """Close and clean up resources."""
        if hasattr(self.llm, "close"):
            self.llm.close()
