from app.config.config_env import LLM_MAX_TOKEN


class TokenUtils:
    def __init__(self, model, max_tokens: int = LLM_MAX_TOKEN):
        self.model = model
        self.max_tokens = max_tokens

    def trim_history_to_fit_tokens(self, conversation_history: list) -> list:
        """Trim the conversation history to ensure the total token count fits within max_tokens."""
        total_tokens = sum(
            self.count_tokens(message.content)
            for message in conversation_history
        )

        while total_tokens > self.max_tokens:
            conversation_history.pop(0)  # Remove the oldest message
            total_tokens = sum(
                self.count_tokens(message.content)
                for message in conversation_history
            )

        return conversation_history

    def count_tokens(
        self, text: str, add_bos: bool = True, special: bool = False
    ) -> int:
        """Estimate the number of tokens in a given text using the model's tokenizer."""
        if isinstance(text, str):
            text = text.encode('utf-8')

        tokens = self.model.tokenizer(text, add_bos=add_bos, special=special)
        return len(tokens)
