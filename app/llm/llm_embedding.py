import logging
import os
from typing import Optional, List

from llama_cpp import Llama


class LLMEmbedding:
    def __init__(self):
        """Initialize LLMEmbedding with embedding mode enabled."""
        base_model_dir = (
            "./model_cache/models--ThanhTranVisma--Llama3.1-8B-blueVi-GPT"
        )
        model_path = self.get_model_path(base_model_dir)
        self.llm = self.load_embedding_model(model_path)

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
            raise ValueError("Model file not found in snapshots directory")

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
