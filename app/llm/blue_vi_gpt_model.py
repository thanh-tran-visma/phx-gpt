import logging
from typing import List

from llama_cpp import (
    Llama,
    ChatCompletionRequestUserMessage,
)
from app.config.config_env import MODEL_NAME, HF_TOKEN, GGUF_MODEL
from app.model import Message
from app.schemas import GptResponseSchema


class BlueViGptModel:
    def __init__(self):
        """Initialize BlueViGptModel with main model and embedding model."""
        self.llm = self.load_model()

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

    def tokenizer(
        self, text: bytes, add_bos: bool, special: bool
    ) -> List[int]:
        return self.llm.tokenize(text, add_bos, special)

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
                content="Sorry, something went wrong while generating a response. Please retry with shorter prompt"
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
                content="Sorry, something went wrong while generating a response. Please retry with shorter prompt"
            )

    def check_for_personal_data(self, prompt: str) -> bool:
        """Detect personal data in the prompt."""
        instruction = (
            f"Detect personal data:\n{prompt}\n" f"return True or False"
        )
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
                # Extract the content from the model's response
                model_response = choices[0]["message"]["content"]

                # The model will return a direct True/False response
                if "True" in model_response:
                    return True
                elif "False" in model_response:
                    return False
                else:
                    logging.warning("Model response is unclear.")
                    return False
            else:
                logging.warning("Model did not return a valid response.")
                return False

        except Exception as e:
            logging.error(f"Error checking for personal data: {e}")
            return False

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
