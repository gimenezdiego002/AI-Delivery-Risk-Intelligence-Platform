"""Validated runtime configuration for the FastAPI backend.

Secrets are read from the process environment (with local ``.env`` support)
and represented as ``SecretStr`` values so accidental string formatting does
not reveal them. Production validation fails at startup when security or
provider configuration is incomplete.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


def _read_bool(name: str, default: bool) -> bool:
    """Parse one conventional environment boolean or fail clearly."""
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of true/false, 1/0, yes/no, or on/off."
    )


def _read_int(name: str, default: int) -> int:
    """Parse an integer environment variable with its name in any error."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _read_float(name: str, default: float) -> float:
    """Parse a floating-point environment variable with a safe error."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc


def _optional_secret(name: str) -> SecretStr | None:
    """Return a non-empty secret without ever placing its value in an error."""
    value = os.getenv(name)
    return SecretStr(value) if value and value.strip() else None


def _csv_values(name: str) -> tuple[str, ...]:
    """Parse a comma-separated list while removing empty entries."""
    raw = os.getenv(name, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


class RuntimeSettings(BaseModel):
    """Typed, immutable configuration shared by the production API."""

    model_config = ConfigDict(frozen=True)

    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    llm_provider: Literal["openai", "gemini"] = "openai"
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    gemini_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_json_retry_limit: int = Field(default=1, ge=0, le=2)
    llm_max_tool_calls: int = Field(default=3, ge=1, le=10)

    api_auth_enabled: bool = False
    api_key: SecretStr | None = None

    rate_limit_enabled: bool = False
    rate_limit_requests: int = Field(default=60, ge=1, le=100_000)
    rate_limit_llm_requests: int = Field(default=10, ge=1, le=100_000)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=86_400)

    cors_allowed_origins: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_runtime_contract(self) -> "RuntimeSettings":
        """Reject unsafe or incomplete production configuration."""
        if not self.llm_model.strip():
            raise ValueError("LLM_MODEL must not be empty.")
        if "*" in self.cors_allowed_origins:
            raise ValueError(
                "CORS_ALLOWED_ORIGINS must contain explicit origins, not '*'."
            )
        if self.api_auth_enabled and self.api_key is None:
            raise ValueError("API_KEY is required when API_AUTH_ENABLED=true.")
        if self.app_env == "production":
            if not self.api_auth_enabled:
                raise ValueError(
                    "API_AUTH_ENABLED must be true when APP_ENV=production."
                )
            provider_secret = (
                self.openai_api_key
                if self.llm_provider == "openai"
                else self.gemini_api_key
            )
            if provider_secret is None:
                variable = (
                    "OPENAI_API_KEY"
                    if self.llm_provider == "openai"
                    else "GEMINI_API_KEY"
                )
                raise ValueError(
                    f"{variable} is required for the selected production provider."
                )
        return self

    @property
    def docs_enabled(self) -> bool:
        """Expose interactive API schemas only outside production."""
        return self.app_env != "production"

    def public_summary(self) -> dict[str, object]:
        """Return non-secret settings safe for startup logs and diagnostics."""
        return {
            "app_env": self.app_env,
            "log_level": self.log_level,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "llm_max_retries": self.llm_max_retries,
            "llm_json_retry_limit": self.llm_json_retry_limit,
            "llm_max_tool_calls": self.llm_max_tool_calls,
            "api_auth_enabled": self.api_auth_enabled,
            "rate_limit_enabled": self.rate_limit_enabled,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_llm_requests": self.rate_limit_llm_requests,
            "rate_limit_window_seconds": self.rate_limit_window_seconds,
            "cors_origin_count": len(self.cors_allowed_origins),
            "docs_enabled": self.docs_enabled,
        }


@lru_cache(maxsize=1)
def get_settings() -> RuntimeSettings:
    """Load and validate settings once per process."""
    load_dotenv(override=False)
    return RuntimeSettings(
        app_env=os.getenv("APP_ENV", "development").strip().lower(),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        llm_provider=os.getenv("LLM_PROVIDER", "openai").strip().lower(),
        openai_api_key=_optional_secret("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        gemini_api_key=_optional_secret("GEMINI_API_KEY"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
        llm_temperature=_read_float("LLM_TEMPERATURE", 0.0),
        llm_timeout_seconds=_read_float("LLM_TIMEOUT_SECONDS", 30.0),
        llm_max_retries=_read_int("LLM_MAX_RETRIES", 2),
        llm_json_retry_limit=_read_int("LLM_JSON_RETRY_LIMIT", 1),
        llm_max_tool_calls=_read_int("LLM_MAX_TOOL_CALLS", 3),
        api_auth_enabled=_read_bool("API_AUTH_ENABLED", False),
        api_key=_optional_secret("API_KEY"),
        rate_limit_enabled=_read_bool("RATE_LIMIT_ENABLED", False),
        rate_limit_requests=_read_int("RATE_LIMIT_REQUESTS", 60),
        rate_limit_llm_requests=_read_int("RATE_LIMIT_LLM_REQUESTS", 10),
        rate_limit_window_seconds=_read_int("RATE_LIMIT_WINDOW_SECONDS", 60),
        cors_allowed_origins=_csv_values("CORS_ALLOWED_ORIGINS"),
    )
