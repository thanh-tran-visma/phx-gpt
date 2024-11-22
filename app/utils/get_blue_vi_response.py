import logging
from typing import List, Tuple, Optional
from llama_cpp import (
    Llama,
    llama_grammar,
)
from starlette.concurrency import run_in_threadpool

from app.types.enum.gpt import Role


def get_operation_format(conversation_history: List[Tuple[str, str]]) -> List:
    formatted_messages = []

    for conversation in conversation_history:
        if len(conversation) != 2:
            logging.error(f"Invalid tuple: {conversation}")
            continue  # Skip invalid tuples
        role, message = conversation
        if role == Role.USER.value:
            formatted_messages.append(
                {'role': Role.USER.value, 'content': message}
            )
        elif role == Role.ASSISTANT.value:
            formatted_messages.append(
                {'role': Role.ASSISTANT.value, 'content': message}
            )
        elif role == Role.SYSTEM.value:
            formatted_messages.append(
                {'role': Role.SYSTEM.value, 'content': message}
            )

    return formatted_messages


async def get_blue_vi_response(
    llm: Llama,
    conversation_history: List[Tuple[str, str]],
    grammar: Optional[llama_grammar] = None,
) -> dict:
    """
    Get the model's response to the conversation history.

    Args:
    - llm (Llama): The Llama model.
    - conversation_history (List[Tuple[str, str]]): A list of tuples containing the role ('user', 'assistant', 'system')
      and message string.

    Returns:
    - dict: The model's response as a dictionary.
    """
    try:
        # Format the conversation history into the appropriate format
        messages = get_operation_format(conversation_history)
        logging.info(messages)
        # Use threadpool to run the model's chat completion
        response = await run_in_threadpool(
            lambda: llm.create_chat_completion(
                messages=messages, grammar=grammar
            )
        )
        return response
    except Exception as e:
        logging.error(f"Error communicating with the model: {e}")
        return {}
