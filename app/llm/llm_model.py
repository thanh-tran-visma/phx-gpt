from llama_cpp import Llama
from huggingface_hub import login
from app.config.env_config import MODEL_NAME, HF_TOKEN


class BlueViGptModel:
    def __init__(self):
        self.llm = self.load_model()

    def load_model(self):
        model_cache_dir = "./model_cache"
        if HF_TOKEN:
            login(HF_TOKEN)
        else:
            raise ValueError("HF_TOKEN environment variable is not set.")

        llm = Llama.from_pretrained(
            repo_id=MODEL_NAME, filename="unsloth.Q8_0.gguf", cache_dir=model_cache_dir
        )
        return llm

    def get_response(self, conversation_history):
        response = self.llm.create_chat_completion(messages=conversation_history)
        if response.get("choices"):
            message_content = response["choices"][0]["message"]["content"]
            return message_content
        else:
            return "Sorry, I couldn't generate a response."

    def get_anonymized_message(self, user_message):
        instruction = "Anonymize the data:\n" f"{user_message}\n"

        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": instruction}]
        )

        if response.get("choices"):
            message_content = response["choices"][0]["message"]["content"]
            return message_content
        else:
            return "Sorry, I couldn't generate an anonymized response.", 0, 0
