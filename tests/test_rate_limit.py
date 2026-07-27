"""Deterministic tests for the single-process rate limiter."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.config import RuntimeSettings
from src.api.rate_limit import InMemoryRateLimitMiddleware
from src.api.request_context import RequestContextMiddleware


def _client(
    *,
    enabled: bool = True,
    standard_limit: int = 2,
    llm_limit: int = 1,
) -> TestClient:
    settings = RuntimeSettings(
        app_env="test",
        rate_limit_enabled=enabled,
        rate_limit_requests=standard_limit,
        rate_limit_llm_requests=llm_limit,
        rate_limit_window_seconds=60,
    )
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/orders/test/risk")
    def inference():
        return {"ok": True}

    @app.post("/agent/query")
    def agent():
        return {"ok": True}

    logger = logging.getLogger("rate-limit-test")
    app.add_middleware(
        InMemoryRateLimitMiddleware,
        settings=settings,
        logger=logger,
        clock=lambda: 100.0,
    )
    app.add_middleware(
        RequestContextMiddleware,
        settings=settings,
        logger=logger,
    )
    return TestClient(app)


def test_standard_limit_returns_429_and_retry_after() -> None:
    client = _client()
    assert client.get("/orders/test/risk").status_code == 200
    assert client.get("/orders/test/risk").status_code == 200
    response = client.get("/orders/test/risk")
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limit_error"
    assert response.headers["Retry-After"] == "60"
    assert "X-Request-ID" in response.headers


def test_llm_bucket_has_a_separate_lower_limit() -> None:
    client = _client()
    assert client.post("/agent/query").status_code == 200
    assert client.post("/agent/query").status_code == 429
    assert client.get("/orders/test/risk").status_code == 200


def test_health_is_exempt_from_rate_limit() -> None:
    client = _client(standard_limit=1, llm_limit=1)
    assert all(client.get("/health").status_code == 200 for _ in range(5))


def test_disabled_rate_limit_allows_requests() -> None:
    client = _client(enabled=False, standard_limit=1)
    assert all(
        client.get("/orders/test/risk").status_code == 200 for _ in range(3)
    )
