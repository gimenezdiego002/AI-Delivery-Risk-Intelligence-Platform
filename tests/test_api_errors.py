"""Deterministic tests for the Phase 10 public error taxonomy."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app, raise_server_exceptions=False)


def test_request_validation_uses_safe_taxonomy_and_request_id() -> None:
    response = client.post(
        "/agent/query",
        json={},
        headers={"X-Request-ID": "validation-test"},
    )
    payload = response.json()
    assert response.status_code == 422
    assert payload["error"]["code"] == "validation_error"
    assert payload["request_id"] == "validation-test"
    assert "detail" not in payload


def test_existing_order_not_found_code_is_preserved() -> None:
    response = client.get(
        "/orders/ffffffffffffffffffffffffffffffff/risk"
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "order_not_found"


def test_plain_router_error_is_safe_and_categorized() -> None:
    with patch(
        "src.api.main.run_agent",
        side_effect=RuntimeError("secret provider diagnostic"),
    ):
        response = client.post("/agent/query", json={"query": "test"})
    rendered = response.text
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "secret provider diagnostic" not in rendered
    assert "traceback" not in rendered.lower()
