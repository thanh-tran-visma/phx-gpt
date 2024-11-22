from typing import List

from app.model import Message


def convert_conversation_history_to_tuples(
    conversation_history: List[Message],
) -> List[tuple[str, str]]:
    """Converts the conversation history to a list of tuples (role, content), ensuring correct order."""
    # Ensure conversation is ordered by timestamp, assuming created_at exists
    sorted_history = sorted(
        conversation_history, key=lambda msg: msg.created_at
    )
    return [(message.role, message.content) for message in sorted_history]
