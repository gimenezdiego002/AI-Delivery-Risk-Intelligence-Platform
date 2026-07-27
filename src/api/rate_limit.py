"""Lightweight in-memory rate limiting for a single API process."""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict, deque
from time import monotonic
from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.api.config import RuntimeSettings
from src.api.errors import AppError, ErrorCategory, app_error_response


_EXEMPT_PATHS = {"/health", "/ready"}


class InMemoryRateLimitMiddleware(BaseHTTPMiddleware):
    """Apply fixed-window limits per client and endpoint cost class.

    This is intentionally a one-process implementation. Multiple workers,
    containers, or replicas would each maintain independent counters; a shared
    store such as Redis is required for globally consistent enforcement.
    """

    def __init__(
        self,
        app,
        *,
        settings: RuntimeSettings,
        logger,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        super().__init__(app)
        self.settings = settings
        self.logger = logger
        self.clock = clock
        self._requests: defaultdict[
            tuple[str, str], deque[float]
        ] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if (
            not self.settings.rate_limit_enabled
            or request.url.path in _EXEMPT_PATHS
        ):
            return await call_next(request)

        bucket = (
            "llm" if request.url.path.startswith("/agent/") else "standard"
        )
        limit = (
            self.settings.rate_limit_llm_requests
            if bucket == "llm"
            else self.settings.rate_limit_requests
        )
        client = request.client.host if request.client else "unknown"
        now = self.clock()
        window = self.settings.rate_limit_window_seconds
        key = (client, bucket)

        async with self._lock:
            timestamps = self._requests[key]
            cutoff = now - window
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if len(timestamps) >= limit:
                retry_after = max(
                    1,
                    math.ceil(timestamps[0] + window - now),
                )
                return app_error_response(
                    AppError(
                        ErrorCategory.RATE_LIMIT,
                        429,
                        retry_after=retry_after,
                    ),
                    logger=self.logger,
                    app_env=self.settings.app_env,
                )
            timestamps.append(now)

        return await call_next(request)
