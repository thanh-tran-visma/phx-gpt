from app.types.enum.http_status import HTTPStatus


class ResponseUtils:
    @staticmethod
    def success_response(bot_response, conversation_order: int):
        return {
            "status": bot_response.status,
            "response": bot_response.content,
            "conversation_order": conversation_order,
            "dynamic_json": bot_response.dynamic_json,
            "type": bot_response.type,
        }

    @staticmethod
    def error_response(status: HTTPStatus, message: str):
        return {"status": status.value, "response": message}
