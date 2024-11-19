import logging

from app.config.config_env import LLM_MAX_TOKEN, MAX_HISTORY_WINDOW_SIZE


class TokenUtils:
    def __init__(self, model):
        self.model = model
        self.max_tokens = LLM_MAX_TOKEN
        self.history_window_size = MAX_HISTORY_WINDOW_SIZE
        self.tokenizer = model.tokenizer

    def trim_history_to_fit_tokens(self, conversation_history: list) -> list:
        """Trim the conversation history based on max tokens and window size."""
        # Limit history to the maximum window size
        conversation_history = conversation_history[
            -self.history_window_size :
        ]
        total_tokens = sum(
            self.count_tokens(message.content)
            for message in conversation_history
        )

        while total_tokens > self.max_tokens:
            removed_message = conversation_history.pop(0)
            total_tokens -= self.count_tokens(removed_message.content)

        return conversation_history

    def count_tokens(
        self, text: str, add_bos: bool = True, special: bool = False
    ) -> int:
        """Estimate the number of tokens in a given text using LlamaTokenizer."""
        if isinstance(text, str):
            text = text.encode('utf-8')

        try:
            tokens = self.tokenizer.tokenize(
                text, add_bos=add_bos, special=special
            )
            return len(tokens)
        except RuntimeError as e:
            logging.error(f"Tokenization failed for text: {text[:50]}...: {e}")
            return 0
