import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.api import router
from app.middleware import CustomMiddleware
from app.types.enum.http_status import HTTPStatus


@pytest.fixture
def client():
    app = FastAPI()
    # noinspection PyTypeChecker
    app.add_middleware(CustomMiddleware)
    app.include_router(router)
    with TestClient(app) as c:
        yield c


@patch("app.auth.Auth.validate_token")
def test_auth_valid_token(mock_validate_token, client):
    mock_validate_token.return_value = True
    headers = {"Authorization": "Bearer mocked_token"}
    response = client.get("/auth", headers=headers)
    assert response.status_code == HTTPStatus.OK.value


@patch("app.auth.Auth.validate_token")
def test_auth_invalid_token(mock_validate_token, client):
    mock_validate_token.return_value = False
    headers = {"Authorization": "Bearer mocked_token"}
    response = client.get("/auth", headers=headers)
    assert response.status_code == HTTPStatus.UNAUTHORIZED.value
    assert response.json() == {"detail": "Not authenticated: Invalid token"}


def test_auth_missing_token(client):
    response = client.get("/auth")
    assert response.status_code == HTTPStatus.UNAUTHORIZED.value
    assert response.json() == {
        "detail": "Unauthenticated: Missing Authorization header"
    }


def test_auth_invalid_format(client):
    headers = {"Authorization": "Basic mocked_token"}
    response = client.get("/auth", headers=headers)
    assert response.status_code == HTTPStatus.UNAUTHORIZED.value
    assert response.json() == {
        "detail": "Not authenticated: Invalid Authorization format"
    }


def test_auth_token_missing_in_format(client):
    headers = {"Authorization": "Bearer "}
    response = client.get("/auth", headers=headers)
    assert response.status_code == HTTPStatus.UNAUTHORIZED.value
    assert response.json() == {"detail": "Not authenticated: Token is missing"}
