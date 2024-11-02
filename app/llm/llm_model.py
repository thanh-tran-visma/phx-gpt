import logging
from typing import List, Optional

from llama_cpp import (
    Llama,
    ChatCompletionRequestUserMessage,
)
from huggingface_hub import login
from app.config.config_env import MODEL_NAME, HF_TOKEN, GGUF_MODEL
from app.llm.llm_embedding import LLMEmbedding
from app.model import Message
from app.schemas import GptResponseSchema


class BlueViGptModel:
    def __init__(self):
        """Initialize BlueViGptModel with main model and embedding model."""
        self.llm = self.load_model()
        self.embedding_model = LLMEmbedding()

    @staticmethod
    def load_model() -> Llama:
        """Load the main Llama model from Hugging Face."""
        model_cache_dir = "./model_cache"
        if HF_TOKEN:
            login(HF_TOKEN)
        else:
            raise ValueError("HF_TOKEN environment variable is not set.")

        try:
            llm = Llama.from_pretrained(
                repo_id=MODEL_NAME,
                filename=GGUF_MODEL,
                cache_dir=model_cache_dir,
            )
            return llm
        except Exception as e:
            logging.error(f"Error loading the model: {e}")
            raise

    def embed(self, text: str) -> Optional[List[float]]:
        """Generate embeddings for the provided text."""
        return self.embedding_model.embed(text)

    def get_chat_response(
        self, conversation_history: List[Message]
    ) -> GptResponseSchema:
        """Generate a response from the model based on conversation history."""
        try:
            # Prepare messages for the model
            mapped_messages: List[ChatCompletionRequestUserMessage] = [
                ChatCompletionRequestUserMessage(
                    role=msg.role, content=msg.content
                )
                for msg in conversation_history
            ]

            # Get response from LLM
            response = self.llm.create_chat_completion(
                messages=mapped_messages
            )
            choices = response.get("choices")

            if isinstance(choices, list) and len(choices) > 0:
                message_content = choices[0]["message"]["content"]
                return GptResponseSchema(content=message_content)

            return GptResponseSchema(
                content="Sorry, I couldn't generate a response."
            )

        except ValueError as ve:
            logging.error(f"Value error in chat response generation: {ve}")
            return GptResponseSchema(
                content="Sorry, there was a value error while generating a response."
            )
        except TypeError as te:
            logging.error(f"Type error in chat response generation: {te}")
            return GptResponseSchema(
                content="Sorry, there was a type error while generating a response."
            )
        except KeyError as ke:
            logging.error(f"Key error in chat response generation: {ke}")
            return GptResponseSchema(
                content="Sorry, there was a key error while generating a response."
            )
        except Exception as e:
            logging.error(
                f"Unexpected error while generating chat response: {e}"
            )
            return GptResponseSchema(
                content="Sorry, something went wrong while generating a response."
            )

    def get_anonymized_message(self, user_message: str) -> GptResponseSchema:
        """Anonymize the user message."""
        instruction = f"Anonymize the data:\n{user_message}\n"

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    ChatCompletionRequestUserMessage(
                        role="user", content=instruction
                    )
                ]
            )
            choices = response.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                message_content = choices[0]["message"]["content"]
                return GptResponseSchema(content=message_content)
            else:
                return GptResponseSchema(
                    content="Sorry, I couldn't generate an anonymized response."
                )

        except Exception as e:
            logging.error(f"Error generating anonymized message: {e}")
            return GptResponseSchema(
                content="Sorry, an error occurred while generating an anonymized response."
            )
