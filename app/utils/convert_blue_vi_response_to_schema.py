from app.schemas import GptResponseSchema
from app.types.enum.http_status import HTTPStatus


def convert_blue_vi_response_to_schema(response: str) -> GptResponseSchema:
    """
    Utility function to process the model's response.

    Args:
        response (str): The raw response from the model.

    Returns:
        GptResponseSchema: A schema with the model's response or error message.
    """
    if response:
        return GptResponseSchema(status=HTTPStatus.OK.value, content=response)
    return GptResponseSchema(
        status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        content="Sorry, I couldn't generate a response.",
    )
