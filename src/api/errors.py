"""Safe public errors and internal Phase 10 error categorization."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from fastapi.responses import JSONResponse

from src.api.logging_config import (
    log_event,
    request_id_context,
    trace_id_context,
)


class ErrorCategory(StrEnum):
    """Stable categories used by logs and new infrastructure errors."""

    VALIDATION = "validation_error"
    AUTHENTICATION = "authentication_error"
    RATE_LIMIT = "rate_limit_error"
    ORDER_NOT_FOUND = "order_not_found"
    SELLER_NOT_FOUND = "seller_not_found"
    TOOL_EXECUTION = "tool_execution_error"
    MODEL_LOADING = "model_loading_error"
    LLM_TIMEOUT = "llm_timeout"
    LLM_RATE_LIMIT = "llm_rate_limit"
    LLM_INVALID_RESPONSE = "llm_invalid_response"
    LLM_PROVIDER = "llm_provider_error"
    AGENT_MAX_STEPS = "agent_max_steps_reached"
    INTERNAL = "internal_error"


PUBLIC_MESSAGES: dict[ErrorCategory, str] = {
    ErrorCategory.VALIDATION: "The request is invalid.",
    ErrorCategory.AUTHENTICATION: "A valid API key is required.",
    ErrorCategory.RATE_LIMIT: "The request limit has been exceeded.",
    ErrorCategory.ORDER_NOT_FOUND: "The requested order was not found.",
    ErrorCategory.SELLER_NOT_FOUND: "The requested seller was not found.",
    ErrorCategory.TOOL_EXECUTION: "The deterministic tool could not complete.",
    ErrorCategory.MODEL_LOADING: "Required model artifacts are unavailable.",
    ErrorCategory.LLM_TIMEOUT: "The language-model provider timed out.",
    ErrorCategory.LLM_RATE_LIMIT: "The language-model provider is rate limited.",
    ErrorCategory.LLM_INVALID_RESPONSE: (
        "The language-model provider returned an invalid structured response."
    ),
    ErrorCategory.LLM_PROVIDER: (
        "The language-model provider could not complete the request."
    ),
    ErrorCategory.AGENT_MAX_STEPS: "The agent reached its maximum step limit.",
    ErrorCategory.INTERNAL: "The service could not complete the request.",
}


class AppError(RuntimeError):
    """Internal exception with a safe public category and HTTP status."""

    def __init__(
        self,
        category: ErrorCategory,
        status_code: int,
        *,
        public_message: str | None = None,
        retry_after: int | None = None,
        internal_context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(public_message or PUBLIC_MESSAGES[category])
        self.category = category
        self.status_code = status_code
        self.public_message = public_message or PUBLIC_MESSAGES[category]
        self.retry_after = retry_after
        self.internal_context = internal_context or {}


def tool_error_category(code: str) -> ErrorCategory:
    """Map existing deterministic-tool codes without changing tool math."""
    if code == "order_not_found":
        return ErrorCategory.ORDER_NOT_FOUND
    if code == "seller_not_found":
        return ErrorCategory.SELLER_NOT_FOUND
    if code in {"artifact_error", "feature_contract_mismatch"}:
        return ErrorCategory.MODEL_LOADING
    if code in {
        "missing_feature_columns",
        "missing_feature_values",
        "duplicate_order",
        "invalid_top_n",
    }:
        return ErrorCategory.VALIDATION
    return ErrorCategory.TOOL_EXECUTION


def app_error_response(
    error: AppError,
    *,
    logger,
    app_env: str,
) -> JSONResponse:
    """Log internal category/context and return only safe public data."""
    log_event(
        logger,
        "warning" if error.status_code < 500 else "error",
        "application_error",
        app_env=app_env,
        status_code=error.status_code,
        error_category=error.category.value,
    )
    payload: dict[str, Any] = {
        "ok": False,
        "error": {
            "code": error.category.value,
            "message": error.public_message,
        },
    }
    request_id = request_id_context.get()
    trace_id = trace_id_context.get()
    if request_id:
        payload["request_id"] = request_id
    if trace_id:
        payload["trace_id"] = trace_id
    headers = (
        {"Retry-After": str(error.retry_after)}
        if error.retry_after is not None
        else None
    )
    return JSONResponse(
        status_code=error.status_code,
        content=payload,
        headers=headers,
    )
