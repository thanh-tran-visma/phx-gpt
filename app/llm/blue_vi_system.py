import os
import logging
from pathlib import Path
from typing import Optional, List

from langchain_core.callbacks import (
    CallbackManager,
    StreamingStdOutCallbackHandler,
)
from llama_cpp import Llama, LlamaTokenizer
from app.config.config_env import (
    MODEL_NAME,
    HF_TOKEN,
    GGUF_MODEL,
    LLM_MAX_TOKEN,
)
from app.llm.blue_vi_assistant import BlueViGptAssistant
from app.schemas import GptResponseSchema
from app.model import Message
from app.types.enum.gpt import Role
from app.types.enum.http_status import HTTPStatus
from app.types.enum.instruction.blue_vi_gpt_instruction_enum import (
    BlueViInstructionEnum,
)
from app.utils import (
    get_blue_vi_response,
    convert_blue_vi_response_to_schema,
    convert_conversation_history_to_tuples,
)


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
    def find_gguf_file(model_cache_dir: str) -> Optional[str]:
        """Search for the GGUF model file in the model cache directory."""
        snapshot_path = Path(model_cache_dir)
        try:
            for root, _, files in os.walk(snapshot_path):
                for file in files:
                    if file == GGUF_MODEL:
                        logging.info(
                            f"Found GGUF model at: {os.path.join(root, file)}"
                        )
                        return os.path.join(root, file)
        except Exception as e:
            logging.error(f"Error while searching for GGUF model: {e}")
        return None

    @staticmethod
    def load_model() -> Llama:
        """Load the main Llama model, checking for local cache first."""
        model_cache_dir = "./model_cache"
        gguf_path = BlueViGptModel.find_gguf_file(model_cache_dir)

        if not gguf_path or not os.path.exists(gguf_path):
            logging.warning(
                f"GGUF model not found locally. Attempting to load from {MODEL_NAME}."
            )
            try:
                return Llama.from_pretrained(
                    repo_id=MODEL_NAME,
                    filename=GGUF_MODEL,
                    cache_dir=model_cache_dir,
                    token=HF_TOKEN,
                    max_tokens=LLM_MAX_TOKEN,
                )
            except Exception as e:
                logging.error(f"Error loading model from Hugging Face: {e}")
                raise

        try:
            callback_manager = CallbackManager(
                [StreamingStdOutCallbackHandler()]
            )
            llm = Llama(
                model_path=gguf_path,
                callback_manager=callback_manager,
                verbose=True,
                max_tokens=LLM_MAX_TOKEN,
            )
            logging.info("Model loaded successfully from local GGUF file.")
            return llm
        except Exception as e:
            logging.error(
                f"Failed to initialize Llama model with GGUF path: {e}"
            )
            raise

    @property
    def tokenizer(self) -> LlamaTokenizer:
        """Return the Llama tokenizer for this model."""
        try:
            return LlamaTokenizer(self.llm)
        except Exception as e:
            logging.error(f"Failed to initialize tokenizer: {e}")
            raise

    def close(self):
        """Close and clean up resources."""
        if hasattr(self.llm, "close"):
            try:
                self.llm.close()
                logging.info("LLM resources closed successfully.")
            except Exception as e:
                logging.error(f"Error during LLM resource cleanup: {e}")

    async def generate_user_response_with_custom_instruction(
        self,
        instruction: Optional[str] = None,
    ) -> GptResponseSchema:
        """Generate a response from the model based on conversation history for the user role, optionally with a custom instruction."""
        try:
            # Use the provided instruction or fall back to the default system instruction
            system_instruction = (
                instruction
                if instruction
                else BlueViInstructionEnum.BLUE_VI_SYSTEM_DEFAULT_INSTRUCTION.value
            )

            # Generate the response
            response = await get_blue_vi_response(
                self.llm, [(Role.SYSTEM.value, system_instruction)]
            )

            # Convert and return the response as GptResponseSchema
            return convert_blue_vi_response_to_schema(response)

        except Exception as e:
            logging.error(
                f"Unexpected error while generating chat response: {e}"
            )
            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content="Sorry, something went wrong while generating a response.",
            )
