import logging
import os
from typing import List, Optional

from llama_cpp import (
    Llama,
    ChatCompletionRequestUserMessage,
)
from huggingface_hub import login
from app.config.config_env import MODEL_NAME, HF_TOKEN, GGUF_MODEL
from app.model import Message
from app.types.llm_assistant import GptResponse


class LLMEmbedding:
    def __init__(self):
        """Initialize LLMEmbedding with embedding mode enabled."""
        base_model_dir = (
            "./model_cache/models--ThanhTranVisma--Llama3.1-8B-blueVi-GPT"
        )
        model_path = self.get_model_path(base_model_dir)
        self.llm = self.load_embedding_model(model_path)

    @staticmethod
    def load_embedding_model(model_path: str) -> Llama:
        """Load the Llama model for generating embeddings."""
        try:
            llm = Llama(model_path=model_path, embedding=True, n_ctx=2048)
            return llm
        except Exception as e:
            logging.error(f"Error loading embedding model: {e}")
            raise

    def embed(self, text: str) -> Optional[List[float]]:
        """Generate embeddings for the input text and flatten if nested."""
        try:
            embedding_response = self.llm.embed(text)
            if isinstance(embedding_response, list) and isinstance(
                embedding_response[0], list
            ):
                # Flatten the nested list
                embedding_vector = embedding_response[0]
                if all(isinstance(x, float) for x in embedding_vector):
                    return embedding_vector
            elif isinstance(embedding_response, list) and all(
                isinstance(x, float) for x in embedding_response
            ):
                return embedding_response

            logging.error(
                f"Invalid embedding vector received for text '{text}': {embedding_response}"
            )
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")

        return None

    @staticmethod
    def get_model_path(base_dir: str) -> str:
        """Find and return the model path dynamically."""
        snapshots_dir = os.path.join(base_dir, "snapshots")
        try:
            # Get the first subdirectory inside snapshots (assumes only one hash folder exists)
            hash_folder = next(
                os.path.join(snapshots_dir, d)
                for d in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, d))
            )
            # Find the model file in the hash directory
            model_file = next(
                f for f in os.listdir(hash_folder) if f.endswith('.gguf')
            )
            return os.path.join(hash_folder, model_file)
        except StopIteration:
            logging.error("Model file not found in snapshots directory")
            raise ValueError("Model file not found in snapshots directory")


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
        """Generate embeddings using the embedding model."""
        return self.get_embedding(text)

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embeddings using the embedding model."""
        embedding_vector = self.embedding_model.embed(text)

        if embedding_vector:
            return embedding_vector

        return None

    def get_chat_response(
        self, conversation_history: List[Message]
    ) -> GptResponse:
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
                return GptResponse(content=message_content)

            return GptResponse("Sorry, I couldn't generate a response.")

        except ValueError as ve:
            logging.error(f"Value error in chat response generation: {ve}")
            return GptResponse(
                "Sorry, there was a value error while generating a response."
            )
        except TypeError as te:
            logging.error(f"Type error in chat response generation: {te}")
            return GptResponse(
                "Sorry, there was a type error while generating a response."
            )
        except KeyError as ke:
            logging.error(f"Key error in chat response generation: {ke}")
            return GptResponse(
                "Sorry, there was a key error while generating a response."
            )
        except Exception as e:
            logging.error(
                f"Unexpected error while generating chat response: {e}"
            )
            return GptResponse(
                "Sorry, something went wrong while generating a response."
            )

    def get_anonymized_message(self, user_message: str) -> GptResponse:
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
                return GptResponse(content=message_content)
            else:
                return GptResponse(
                    "Sorry, I couldn't generate an anonymized response."
                )

        except Exception as e:
            logging.error(f"Error generating anonymized message: {e}")
            return GptResponse(
                "Sorry, an error occurred while generating an anonymized response."
            )
