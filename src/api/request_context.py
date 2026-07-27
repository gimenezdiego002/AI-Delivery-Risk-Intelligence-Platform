"""Request correlation and total-latency middleware."""

from __future__ import annotations

import re
from time import perf_counter
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.api.config import RuntimeSettings
from src.api.logging_config import (
    bind_log_context,
    log_event,
    reset_log_context,
)


_SAFE_CORRELATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def safe_correlation_id(value: str | None) -> str:
    """Accept a bounded safe caller ID or generate an opaque UUID."""
    if value and _SAFE_CORRELATION_ID.fullmatch(value):
        return value
    return str(uuid4())


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach request/trace IDs and emit total request latency."""

    def __init__(
        self,
        app,
        *,
        settings: RuntimeSettings,
        logger,
    ) -> None:
        super().__init__(app)
        self.settings = settings
        self.logger = logger

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = safe_correlation_id(request.headers.get("X-Request-ID"))
        is_agent_request = request.url.path.startswith("/agent/")
        trace_id = (
            safe_correlation_id(request.headers.get("X-Trace-ID"))
            if is_agent_request
            else None
        )
        request.state.request_id = request_id
        request.state.trace_id = trace_id
        tokens = bind_log_context(request_id, trace_id)
        started = perf_counter()

        log_event(
            self.logger,
            "info",
            "request_started",
            app_env=self.settings.app_env,
            http_method=request.method,
            route=request.url.path,
        )
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (perf_counter() - started) * 1_000
            log_event(
                self.logger,
                "error",
                "request_failed",
                app_env=self.settings.app_env,
                http_method=request.method,
                route=request.url.path,
                status_code=500,
                latency_ms=round(elapsed_ms, 3),
                error_category="internal_error",
            )
            raise
        else:
            response.headers["X-Request-ID"] = request_id
            if trace_id is not None:
                response.headers["X-Trace-ID"] = trace_id

            matched_route = request.scope.get("route")
            route_template = getattr(matched_route, "path", request.url.path)
            elapsed_ms = (perf_counter() - started) * 1_000
            log_event(
                self.logger,
                "info" if response.status_code < 500 else "error",
                "request_completed",
                app_env=self.settings.app_env,
                http_method=request.method,
                route=route_template,
                status_code=response.status_code,
                latency_ms=round(elapsed_ms, 3),
            )
            return response
        finally:
            reset_log_context(tokens)
