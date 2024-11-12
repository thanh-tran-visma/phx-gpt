import logging
from llama_cpp import Llama, LlamaTokenizer
from app.config.config_env import MODEL_NAME, HF_TOKEN, GGUF_MODEL
from app.llm.blue_vi_assistant_role import BlueViGptAssistantRole
from app.llm.blue_vi_user_role import BlueViGptUserRole


class BlueViGptModel:
    def __init__(self):
        """Initialize BlueViGptModel with main model and embedding model."""
        self.llm = self.load_model()
        self.user_role = BlueViGptUserRole(self.llm)
        self.assistant_role = BlueViGptAssistantRole(self.llm)

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
