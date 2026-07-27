"""Simple constant-time API-key protection for portfolio deployment."""

from __future__ import annotations

import secrets

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.api.config import RuntimeSettings
from src.api.errors import AppError, ErrorCategory, app_error_response


_PUBLIC_PATHS = {
    "/health",
    "/ready",
    "/docs",
    "/redoc",
    "/openapi.json",
}


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """Protect inference and agent routes when explicitly enabled."""

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
        if (
            not self.settings.api_auth_enabled
            or request.url.path in _PUBLIC_PATHS
        ):
            return await call_next(request)

        supplied = request.headers.get("X-API-Key")
        expected = (
            self.settings.api_key.get_secret_value()
            if self.settings.api_key is not None
            else ""
        )
        if (
            not supplied
            or not expected
            or not secrets.compare_digest(supplied, expected)
        ):
            return app_error_response(
                AppError(ErrorCategory.AUTHENTICATION, 401),
                logger=self.logger,
                app_env=self.settings.app_env,
            )
        return await call_next(request)
