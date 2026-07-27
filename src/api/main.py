"""FastAPI HTTP layer for the delivery-risk intelligence project.

Phase 6 intentionally wraps the already-tested Phase 4 tools and Phase 5
router. It does not retrain models, rebuild datasets, or change tool logic.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.agent.langgraph_agent import run_langgraph_agent
from src.agent.router import (
    LLMInvalidResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    RouterError,
    run_agent,
)
from src.agent.tools import explain_risk, get_seller_history, predict_delay_risk
from src.api.config import get_settings
from src.api.cors import configure_cors
from src.api.errors import (
    AppError,
    ErrorCategory,
    app_error_response,
    tool_error_category,
)
from src.api.logging_config import configure_logging, log_event, query_metadata
from src.api.request_context import RequestContextMiddleware
from src.api.rate_limit import InMemoryRateLimitMiddleware
from src.api.readiness import verify_runtime_readiness
from src.api.security import ApiKeyAuthMiddleware


settings = get_settings()
logger = configure_logging(settings)
app = FastAPI(
    title="AI-Powered Delivery Risk Intelligence Agent",
    description=(
        "HTTP API exposing deterministic delivery-risk tools and the Phase 5 "
        "plain-Python LLM router."
    ),
    version="0.1.0",
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.docs_enabled else None,
    openapi_url="/openapi.json" if settings.docs_enabled else None,
)
app.state.settings = settings
app.state.logger = logger
app.add_middleware(
    ApiKeyAuthMiddleware,
    settings=settings,
    logger=logger,
)
app.add_middleware(
    InMemoryRateLimitMiddleware,
    settings=settings,
    logger=logger,
)
app.add_middleware(
    RequestContextMiddleware,
    settings=settings,
    logger=logger,
)
configure_cors(app, settings)


@app.exception_handler(AppError)
async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
    """Return a stable safe error without exposing internal exception text."""
    return app_error_response(exc, logger=logger, app_env=settings.app_env)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    _request: Request, _exc: RequestValidationError
) -> JSONResponse:
    """Replace FastAPI's detailed validation dump with a stable public error."""
    return app_error_response(
        AppError(ErrorCategory.VALIDATION, 422),
        logger=logger,
        app_env=settings.app_env,
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(
    _request: Request, _exc: Exception
) -> JSONResponse:
    """Prevent stack traces and exception strings from entering API responses."""
    return app_error_response(
        AppError(ErrorCategory.INTERNAL, 500),
        logger=logger,
        app_env=settings.app_env,
    )


class HealthResponse(BaseModel):
    """Service health response that avoids loading model artifacts."""

    status: Literal["ok"]
    model: Literal["logistic_regression"]
    phase: Literal[6]


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Confirm the API process is running without loading model/data files."""
    return HealthResponse(status="ok", model="logistic_regression", phase=6)


class ReadinessResponse(BaseModel):
    """Artifact readiness without any external provider request."""

    status: Literal["ready"]
    model: Literal["logistic_regression"]
    feature_count: int = Field(gt=0)
    llm_checked: Literal[False]


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["service"],
    responses={503: {"model": dict}},
)
def ready() -> ReadinessResponse:
    """Confirm model and feature artifacts can load; never call the LLM."""
    return ReadinessResponse(**verify_runtime_readiness())


class ToolError(BaseModel):
    """Structured tool error returned by deterministic Phase 4 tools."""

    code: str
    message: str


class RiskResponse(BaseModel):
    """Successful late-delivery risk prediction response."""

    ok: Literal[True]
    order_id: str
    late_delivery_probability: float = Field(ge=0, le=1)
    risk_level: Literal["high", "low"]
    model_name: str
    threshold: float = Field(ge=0, le=1)


class ExplanationItem(BaseModel):
    """One feature contribution in an order-specific explanation."""

    feature: str
    actual_value: Any
    direction: Literal["increases_risk", "decreases_risk"]
    signed_log_odds_contribution: float
    approximate_magnitude: float


class ExplanationResponse(BaseModel):
    """Successful deterministic risk explanation response."""

    ok: Literal[True]
    order_id: str
    late_delivery_probability: float = Field(ge=0, le=1)
    risk_level: Literal["high", "low"]
    model_name: str
    explanations: list[ExplanationItem]
    summary: str
    caveat: str


class SellerHistoryResponse(BaseModel):
    """Successful leakage-safe seller-history snapshot response."""

    ok: Literal[True]
    seller_id: str
    as_of_order_id: str
    history_cutoff: str
    historical_order_volume: int = Field(ge=0)
    historical_late_rate: float | None = Field(default=None, ge=0, le=1)
    historical_avg_review_score: float | None = Field(default=None, ge=0, le=5)
    history_source: str
    leakage_rule: str


class AgentQueryRequest(BaseModel):
    """Natural-language query sent to the Phase 5 agent router."""

    query: str = Field(min_length=1)


class AgentQueryResponse(BaseModel):
    """Grounded natural-language answer plus executed tool names."""

    ok: bool
    status: str
    answer: str
    tools_called: list[str]
    tool_call_count: int


class LangGraphAgentQueryResponse(AgentQueryResponse):
    """Inspectable LangGraph response without hidden model reasoning."""

    implementation: Literal["langgraph"]
    trace: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    clarification_message: str | None = None
    error: dict[str, Any] | None = None
    stop_reason: str | None = None


def _status_code_for_tool_error(result: dict[str, Any]) -> int:
    """Map structured tool errors to HTTP client-error status codes."""
    code = result.get("error", {}).get("code", "")
    if code.endswith("_not_found") or code in {"order_not_found", "seller_not_found"}:
        return 404
    return 422


def _tool_error_response(result: dict[str, Any]) -> JSONResponse:
    """Return a structured JSON response without turning tool errors into 500s."""
    code = result.get("error", {}).get("code", "")
    category = tool_error_category(code)
    log_event(
        logger,
        "warning" if _status_code_for_tool_error(result) < 500 else "error",
        "deterministic_tool_error",
        app_env=settings.app_env,
        status_code=_status_code_for_tool_error(result),
        error_category=category.value,
    )
    return JSONResponse(
        status_code=_status_code_for_tool_error(result),
        content=result,
    )


@app.get(
    "/orders/{order_id}/risk",
    response_model=RiskResponse,
    tags=["orders"],
    responses={404: {"model": dict}, 422: {"model": dict}},
)
def get_order_risk(order_id: str) -> RiskResponse | JSONResponse:
    """Return the saved model's structured late-delivery risk prediction."""
    started = perf_counter()
    result = predict_delay_risk(order_id)
    log_event(
        logger,
        "info",
        "deterministic_endpoint_completed",
        app_env=settings.app_env,
        selected_tool="predict_delay_risk",
        latency_ms=round((perf_counter() - started) * 1_000, 3),
        outcome_ok=bool(result.get("ok", False)),
        model_name="logistic_regression",
    )
    if not result["ok"]:
        return _tool_error_response(result)
    return RiskResponse(**result)


@app.get(
    "/orders/{order_id}/explanation",
    response_model=ExplanationResponse,
    tags=["orders"],
    responses={404: {"model": dict}, 422: {"model": dict}},
)
def get_order_explanation(order_id: str) -> ExplanationResponse | JSONResponse:
    """Return deterministic model-feature associations for one order."""
    started = perf_counter()
    result = explain_risk(order_id)
    log_event(
        logger,
        "info",
        "deterministic_endpoint_completed",
        app_env=settings.app_env,
        selected_tool="explain_risk",
        latency_ms=round((perf_counter() - started) * 1_000, 3),
        outcome_ok=bool(result.get("ok", False)),
        model_name="logistic_regression",
    )
    if not result["ok"]:
        return _tool_error_response(result)
    return ExplanationResponse(**result)


@app.get(
    "/sellers/{seller_id}/history",
    response_model=SellerHistoryResponse,
    tags=["sellers"],
    responses={404: {"model": dict}, 422: {"model": dict}},
)
def get_seller_history_endpoint(
    seller_id: str,
) -> SellerHistoryResponse | JSONResponse:
    """Return the Phase 2 leakage-safe seller-history snapshot."""
    started = perf_counter()
    result = get_seller_history(seller_id)
    log_event(
        logger,
        "info",
        "deterministic_endpoint_completed",
        app_env=settings.app_env,
        selected_tool="get_seller_history",
        latency_ms=round((perf_counter() - started) * 1_000, 3),
        outcome_ok=bool(result.get("ok", False)),
    )
    if not result["ok"]:
        return _tool_error_response(result)
    return SellerHistoryResponse(**result)


@app.post(
    "/agent/query",
    response_model=AgentQueryResponse,
    tags=["agent"],
    responses={502: {"model": dict}},
)
def query_agent(request: AgentQueryRequest) -> AgentQueryResponse | JSONResponse:
    """Run the Phase 5 LLM router; this endpoint makes billable API calls."""
    started = perf_counter()
    safe_query = query_metadata(request.query)
    try:
        result = run_agent(request.query)
    except LLMTimeoutError:
        return app_error_response(
            AppError(ErrorCategory.LLM_TIMEOUT, 504),
            logger=logger,
            app_env=settings.app_env,
        )
    except LLMRateLimitError:
        return app_error_response(
            AppError(ErrorCategory.LLM_RATE_LIMIT, 503),
            logger=logger,
            app_env=settings.app_env,
        )
    except LLMInvalidResponseError:
        return app_error_response(
            AppError(ErrorCategory.LLM_INVALID_RESPONSE, 502),
            logger=logger,
            app_env=settings.app_env,
        )
    except (LLMProviderError, RouterError):
        return app_error_response(
            AppError(ErrorCategory.LLM_PROVIDER, 502),
            logger=logger,
            app_env=settings.app_env,
        )

    tools_called = [
        event["tool_name"]
        for event in result.get("trace", [])
        if event.get("event") == "tool_result"
    ]
    log_event(
        logger,
        "info",
        "agent_execution_completed",
        app_env=settings.app_env,
        agent_implementation="plain_python",
        tool_names=tools_called,
        tool_call_count=int(result.get("tool_call_count", len(tools_called))),
        latency_ms=round((perf_counter() - started) * 1_000, 3),
        **safe_query,
    )
    return AgentQueryResponse(
        ok=bool(result.get("ok", False)),
        status=str(result.get("status", "unknown")),
        answer=str(result.get("answer", "")),
        tools_called=tools_called,
        tool_call_count=int(result.get("tool_call_count", len(tools_called))),
    )


@app.post(
    "/agent/langgraph/query",
    response_model=LangGraphAgentQueryResponse,
    tags=["agent"],
    responses={502: {"model": dict}},
)
def query_langgraph_agent(
    request: AgentQueryRequest,
) -> LangGraphAgentQueryResponse | JSONResponse:
    """Run the Phase 9 LangGraph workflow; this makes billable LLM API calls."""
    started = perf_counter()
    safe_query = query_metadata(request.query)
    try:
        result = run_langgraph_agent(request.query)
    except LLMTimeoutError:
        error = AppError(ErrorCategory.LLM_TIMEOUT, 504)
    except LLMRateLimitError:
        error = AppError(ErrorCategory.LLM_RATE_LIMIT, 503)
    except LLMInvalidResponseError:
        error = AppError(ErrorCategory.LLM_INVALID_RESPONSE, 502)
    except (LLMProviderError, RouterError):
        error = AppError(ErrorCategory.LLM_PROVIDER, 502)
    except Exception:
        error = AppError(ErrorCategory.INTERNAL, 500)
    if "error" in locals():
        return app_error_response(
            error,
            logger=logger,
            app_env=settings.app_env,
        )

    tool_results = result.get("tool_results", [])
    tools_called = [
        item["tool_name"]
        for item in tool_results
        if item.get("tool_name")
    ]
    log_event(
        logger,
        "info",
        "agent_execution_completed",
        app_env=settings.app_env,
        agent_implementation="langgraph",
        tool_names=tools_called,
        tool_call_count=int(result.get("tool_call_count", len(tools_called))),
        latency_ms=round((perf_counter() - started) * 1_000, 3),
        stop_reason=result.get("stop_reason"),
        **safe_query,
    )
    return LangGraphAgentQueryResponse(
        ok=bool(result.get("ok", False)),
        status=str(result.get("status", "unknown")),
        answer=str(result.get("answer", "")),
        tools_called=tools_called,
        tool_call_count=int(result.get("tool_call_count", len(tools_called))),
        implementation="langgraph",
        trace=result.get("trace", []),
        tool_results=tool_results,
        clarification_message=result.get("clarification_message"),
        error=result.get("error"),
        stop_reason=result.get("stop_reason"),
    )
