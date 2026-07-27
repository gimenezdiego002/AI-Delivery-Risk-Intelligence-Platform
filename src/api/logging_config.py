"""Safe structured logging for the delivery-risk API.

Production logs are newline-delimited JSON. Development logs are compact text.
Only explicitly allowlisted metadata can be attached, preventing accidental
logging of secrets, authorization headers, raw queries, feature rows, or hidden
model reasoning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any, Literal

from src.api.config import RuntimeSettings


LOGGER_NAME = "delivery_risk"
request_id_context: ContextVar[str | None] = ContextVar(
    "request_id", default=None
)
trace_id_context: ContextVar[str | None] = ContextVar("trace_id", default=None)

_SAFE_EVENT_FIELDS = {
    "http_method",
    "route",
    "status_code",
    "latency_ms",
    "agent_implementation",
    "tool_names",
    "tool_call_count",
    "model_name",
    "error_category",
    "retry_count",
    "provider_call_kind",
    "query_hash",
    "query_length",
    "node",
    "action",
    "selected_tool",
    "outcome_ok",
    "stop_reason",
}


def query_metadata(query: str) -> dict[str, int | str]:
    """Represent a query without retaining its text."""
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return {"query_hash": digest, "query_length": len(query)}


def bind_log_context(
    request_id: str, trace_id: str | None = None
) -> tuple[Token[str | None], Token[str | None]]:
    """Bind safe correlation identifiers to the current async context."""
    return (
        request_id_context.set(request_id),
        trace_id_context.set(trace_id),
    )


def reset_log_context(
    tokens: tuple[Token[str | None], Token[str | None]],
) -> None:
    """Restore the prior request context after middleware completes."""
    request_id_context.reset(tokens[0])
    trace_id_context.reset(tokens[1])


def _base_payload(record: logging.LogRecord) -> dict[str, Any]:
    """Build common machine-readable fields for one log record."""
    payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": record.levelname,
        "event": getattr(record, "event_name", record.getMessage()),
        "request_id": request_id_context.get(),
        "trace_id": trace_id_context.get(),
        "app_env": getattr(record, "app_env", None),
    }
    metadata = getattr(record, "event_metadata", {})
    if isinstance(metadata, dict):
        payload.update(
            {
                key: value
                for key, value in metadata.items()
                if key in _SAFE_EVENT_FIELDS and value is not None
            }
        )
    return payload


class JsonEventFormatter(logging.Formatter):
    """Format one safe event as compact JSON."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            _base_payload(record),
            separators=(",", ":"),
            default=str,
        )


class DevelopmentEventFormatter(logging.Formatter):
    """Format one event as readable local-development text."""

    def format(self, record: logging.LogRecord) -> str:
        payload = _base_payload(record)
        prefix = (
            f"{payload.pop('timestamp')} {payload.pop('level')} "
            f"{payload.pop('event')}"
        )
        details = " ".join(
            f"{key}={json.dumps(value, default=str)}"
            for key, value in payload.items()
            if value is not None
        )
        return f"{prefix} {details}".rstrip()


def configure_logging(settings: RuntimeSettings) -> logging.Logger:
    """Configure the project logger once using the selected environment."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(settings.log_level)
    logger.propagate = False
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonEventFormatter()
        if settings.app_env == "production"
        else DevelopmentEventFormatter()
    )
    logger.addHandler(handler)
    return logger


def log_event(
    logger: logging.Logger,
    level: Literal["debug", "info", "warning", "error", "critical"],
    event: str,
    *,
    app_env: str,
    **metadata: Any,
) -> None:
    """Emit an allowlisted event without accepting arbitrary log fields."""
    safe_metadata = {
        key: value
        for key, value in metadata.items()
        if key in _SAFE_EVENT_FIELDS
    }
    getattr(logger, level)(
        event,
        extra={
            "event_name": event,
            "event_metadata": safe_metadata,
            "app_env": app_env,
        },
    )
