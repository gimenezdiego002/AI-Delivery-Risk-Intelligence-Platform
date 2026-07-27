"""Deterministic tests for Phase 10 runtime configuration."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.api.config import get_settings


def _load_with(environment: dict[str, str]):
    get_settings.cache_clear()
    with (
        patch.dict("os.environ", environment, clear=True),
        patch("src.api.config.load_dotenv"),
    ):
        return get_settings()


def test_development_defaults_are_safe_and_secret_free() -> None:
    settings = _load_with({})
    assert settings.app_env == "development"
    assert settings.api_auth_enabled is False
    assert settings.rate_limit_enabled is False
    assert settings.openai_api_key is None
    assert "api_key" not in settings.public_summary()


def test_authentication_requires_a_key_when_enabled() -> None:
    with pytest.raises(ValidationError, match="API_KEY is required"):
        _load_with({"API_AUTH_ENABLED": "true"})


def test_production_requires_authentication() -> None:
    with pytest.raises(ValidationError, match="API_AUTH_ENABLED must be true"):
        _load_with({"APP_ENV": "production"})


def test_production_requires_selected_provider_secret() -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY is required"):
        _load_with(
            {
                "APP_ENV": "production",
                "API_AUTH_ENABLED": "true",
                "API_KEY": "test-backend-key",
                "LLM_PROVIDER": "openai",
            }
        )


def test_production_settings_validate_without_exposing_secrets() -> None:
    settings = _load_with(
        {
            "APP_ENV": "production",
            "API_AUTH_ENABLED": "true",
            "API_KEY": "test-backend-key",
            "OPENAI_API_KEY": "test-provider-key",
            "LLM_PROVIDER": "openai",
            "CORS_ALLOWED_ORIGINS": "https://demo.example, https://ops.example",
        }
    )
    assert settings.docs_enabled is False
    assert settings.cors_allowed_origins == (
        "https://demo.example",
        "https://ops.example",
    )
    serialized = repr(settings)
    assert "test-backend-key" not in serialized
    assert "test-provider-key" not in serialized


def teardown_module() -> None:
    """Avoid leaking a cached test configuration into later test modules."""
    get_settings.cache_clear()
