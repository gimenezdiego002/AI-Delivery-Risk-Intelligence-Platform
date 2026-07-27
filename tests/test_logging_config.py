"""Tests proving Phase 10 logs are structured and secret-safe."""

from __future__ import annotations

import json
import logging

from src.api.logging_config import (
    JsonEventFormatter,
    bind_log_context,
    log_event,
    query_metadata,
    reset_log_context,
)


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


def test_production_event_is_json_with_correlation_fields() -> None:
    logger = logging.getLogger("phase10-test-json")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = _CaptureHandler()
    handler.setFormatter(JsonEventFormatter())
    logger.addHandler(handler)

    tokens = bind_log_context("request-123", "trace-456")
    try:
        log_event(
            logger,
            "info",
            "request_completed",
            app_env="production",
            http_method="GET",
            route="/health",
            status_code=200,
            latency_ms=12.5,
        )
    finally:
        reset_log_context(tokens)

    payload = json.loads(handler.messages[0])
    assert payload["event"] == "request_completed"
    assert payload["request_id"] == "request-123"
    assert payload["trace_id"] == "trace-456"
    assert payload["latency_ms"] == 12.5


def test_unknown_or_secret_metadata_is_discarded() -> None:
    logger = logging.getLogger("phase10-test-secret")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = _CaptureHandler()
    handler.setFormatter(JsonEventFormatter())
    logger.addHandler(handler)

    log_event(
        logger,
        "info",
        "safe_event",
        app_env="test",
        api_key="must-not-appear",
        authorization="must-not-appear",
        query="full private query",
        status_code=200,
    )

    rendered = handler.messages[0]
    assert "must-not-appear" not in rendered
    assert "full private query" not in rendered
    assert json.loads(rendered)["status_code"] == 200


def test_query_metadata_hashes_instead_of_storing_text() -> None:
    query = "Will my private order be late?"
    metadata = query_metadata(query)
    assert metadata["query_length"] == len(query)
    assert len(str(metadata["query_hash"])) == 16
    assert query not in str(metadata)
