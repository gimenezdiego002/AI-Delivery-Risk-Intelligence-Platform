"""Tests for opt-in explicit-origin CORS behavior."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.config import RuntimeSettings
from src.api.cors import configure_cors


def _app(settings: RuntimeSettings) -> TestClient:
    app = FastAPI()

    @app.get("/test")
    def test_route():
        return {"ok": True}

    configure_cors(app, settings)
    return TestClient(app)


def test_no_origins_adds_no_cors_headers() -> None:
    response = _app(RuntimeSettings()).get(
        "/test", headers={"Origin": "https://demo.example"}
    )
    assert "access-control-allow-origin" not in response.headers


def test_only_explicit_origin_is_allowed() -> None:
    client = _app(
        RuntimeSettings(cors_allowed_origins=("https://demo.example",))
    )
    allowed = client.options(
        "/test",
        headers={
            "Origin": "https://demo.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    denied = client.options(
        "/test",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed.headers["access-control-allow-origin"] == (
        "https://demo.example"
    )
    assert "access-control-allow-origin" not in denied.headers


def test_wildcard_origin_is_rejected() -> None:
    with pytest.raises(ValidationError, match="explicit origins"):
        RuntimeSettings(cors_allowed_origins=("*",))
