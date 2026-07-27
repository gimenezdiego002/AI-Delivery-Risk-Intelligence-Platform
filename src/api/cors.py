"""Opt-in explicit-origin CORS configuration."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import RuntimeSettings


def configure_cors(app: FastAPI, settings: RuntimeSettings) -> bool:
    """Add CORS only when explicit browser origins are configured.

    The current Streamlit client calls FastAPI server-side, so its normal
    architecture needs no browser CORS. Wildcard origins are rejected by
    settings validation.
    """
    if not settings.cors_allowed_origins:
        return False
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "X-API-Key",
            "X-Request-ID",
            "X-Trace-ID",
        ],
    )
    return True
