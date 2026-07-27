"""Deterministic API-key authentication tests."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from src.api.config import RuntimeSettings
from src.api.request_context import RequestContextMiddleware
from src.api.security import ApiKeyAuthMiddleware


def _client(*, enabled: bool) -> TestClient:
    settings = RuntimeSettings(
        app_env="test",
        api_auth_enabled=enabled,
        api_key=SecretStr("correct-test-key") if enabled else None,
    )
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/protected")
    def protected():
        return {"ok": True}

    logger = logging.getLogger("auth-test")
    app.add_middleware(
        ApiKeyAuthMiddleware,
        settings=settings,
        logger=logger,
    )
    app.add_middleware(
        RequestContextMiddleware,
        settings=settings,
        logger=logger,
    )
    return TestClient(app)


def test_authentication_disabled_allows_local_request() -> None:
    assert _client(enabled=False).get("/protected").status_code == 200


def test_valid_api_key_is_accepted() -> None:
    response = _client(enabled=True).get(
        "/protected", headers={"X-API-Key": "correct-test-key"}
    )
    assert response.status_code == 200


def test_missing_api_key_returns_401_without_expected_key() -> None:
    response = _client(enabled=True).get("/protected")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_error"
    assert "correct-test-key" not in response.text


def test_incorrect_api_key_returns_401() -> None:
    response = _client(enabled=True).get(
        "/protected", headers={"X-API-Key": "incorrect"}
    )
    assert response.status_code == 401


def test_health_remains_public() -> None:
    response = _client(enabled=True).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
