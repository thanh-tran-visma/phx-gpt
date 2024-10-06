from enum import Enum

class HTTPStatus(Enum):
    OK = 200
    BAD_REQUEST = 400
    UNPROCESSABLE_ENTITY = 422
    INTERNAL_SERVER_ERROR = 500
