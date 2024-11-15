from app.schemas import GptResponseSchema
from app.types.enum.http_status import HTTPStatus


def process_model_response(response: dict) -> GptResponseSchema:
    """
    Utility function to process the model's response.

    Args:
        response (dict): The raw response from the model.

    Returns:
        GptResponseSchema: A schema with the model's response or error message.
    """
    choices = response.get("choices")
    if isinstance(choices, list) and len(choices) > 0:
        message_content = choices[0]["message"]["content"]
        return GptResponseSchema(
            status=HTTPStatus.OK.value, content=message_content
        )
    return GptResponseSchema(
        status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        content="Sorry, I couldn't generate a response.",
    )
