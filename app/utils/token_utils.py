import logging
from collections import deque

from app.config.config_env import LLM_MAX_TOKEN, MAX_HISTORY_WINDOW_SIZE


class TokenUtils:
    def __init__(self, model):
        self.model = model
        self.max_tokens = LLM_MAX_TOKEN - 50
        self.history_window_size = MAX_HISTORY_WINDOW_SIZE - 50
        self.tokenizer = model.tokenizer

    def trim_history_to_fit_tokens(self, conversation_history: list) -> list:
        """Trim the conversation history based on max tokens and window size."""
        # Limit history to the maximum window size
        conversation_history = deque(
            conversation_history[-self.history_window_size :]
        )

        total_tokens = sum(
            self.count_tokens(message.content)
            for message in conversation_history
        )
        logging.debug(
            f"Initial total tokens: {total_tokens} | Max tokens: {self.max_tokens}"
        )

        while total_tokens > self.max_tokens and conversation_history:
            removed_message = conversation_history.popleft()
            removed_tokens = self.count_tokens(removed_message.content)
            total_tokens -= removed_tokens
            logging.debug(
                f"Removed: {removed_message.content[:50]}... | Removed tokens: {removed_tokens} | Remaining total: {total_tokens}"
            )

        return list(conversation_history)

    def count_tokens(
        self, text: str, add_bos: bool = True, special: bool = False
    ) -> int:
        """Estimate the number of tokens in a given text."""
        if isinstance(text, str):
            text = text.encode('utf-8')
        try:
            tokens = self.tokenizer.tokenize(
                text, add_bos=add_bos, special=special
            )
            logging.info(
                f"Text: {text[:50]}... | Tokens: {tokens} | Count: {len(tokens)}"
            )
            return len(tokens)
        except RuntimeError as e:
            logging.error(f"Tokenization failed for text: {text[:50]}...: {e}")
            return 0
