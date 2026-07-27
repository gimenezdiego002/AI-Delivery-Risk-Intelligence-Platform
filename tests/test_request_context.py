"""Tests for request and agent trace correlation."""

from __future__ import annotations

from unittest.mock import patch
from uuid import UUID

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_accepts_a_safe_request_id() -> None:
    response = client.get("/health", headers={"X-Request-ID": "interview-demo-1"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "interview-demo-1"
    assert "X-Trace-ID" not in response.headers


def test_invalid_request_id_is_replaced_with_uuid() -> None:
    response = client.get(
        "/health",
        headers={"X-Request-ID": "unsafe value with spaces"},
    )
    generated = response.headers["X-Request-ID"]
    assert generated != "unsafe value with spaces"
    assert str(UUID(generated)) == generated


def test_agent_request_returns_request_and_trace_ids() -> None:
    result = {
        "ok": True,
        "status": "completed",
        "answer": "Grounded answer.",
        "tool_call_count": 0,
        "trace": [],
    }
    with patch("src.api.main.run_agent", return_value=result):
        response = client.post(
            "/agent/query",
            json={"query": "test"},
            headers={
                "X-Request-ID": "request-abc",
                "X-Trace-ID": "trace-xyz",
            },
        )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "request-abc"
    assert response.headers["X-Trace-ID"] == "trace-xyz"
