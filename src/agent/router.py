"""Plain-Python LLM router over the deterministic Phase 4 tools.

This module makes routing decisions and orchestrates tool calls. Business logic
remains in ``src.agent.tools``; no agent framework is used.
"""

from __future__ import annotations

import json
import os
import random
import time
from functools import lru_cache
from time import perf_counter
from typing import Any, Literal

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError, model_validator

from src.agent.tools import (
    explain_risk,
    get_seller_history,
    get_similar_past_orders,
    predict_delay_risk,
)
from src.observability import emit_observation, measure_call
from src.api.config import get_settings


TOOL_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "predict_delay_risk",
        "description": (
            "Use when the user asks whether a specific order may be late, its "
            "late-delivery probability, or its high/low risk classification."
        ),
        "required_arguments": {
            "order_id": "Exact order identifier supplied by the user."
        },
    },
    {
        "name": "explain_risk",
        "description": (
            "Use when the user asks why a specific order has its predicted "
            "risk or which model features increased/decreased that risk."
        ),
        "required_arguments": {
            "order_id": "Exact order identifier supplied by the user."
        },
    },
    {
        "name": "get_seller_history",
        "description": (
            "Use when the user asks about a specific seller's historical order "
            "volume, late-delivery rate, review score, or reliability."
        ),
        "required_arguments": {
            "seller_id": "Exact seller identifier supplied by the user."
        },
    },
    {
        "name": "get_similar_past_orders",
        "description": (
            "Use when the user asks for completed orders similar or comparable "
            "to a specific order, including their actual outcomes."
        ),
        "required_arguments": {
            "order_id": "Exact order identifier supplied by the user."
        },
    },
]

TOOL_FUNCTIONS = {
    "predict_delay_risk": predict_delay_risk,
    "explain_risk": explain_risk,
    "get_seller_history": get_seller_history,
    "get_similar_past_orders": get_similar_past_orders,
}


class RouterError(RuntimeError):
    """Raised when the LLM cannot produce a valid routing decision."""


class LLMTimeoutError(RouterError):
    """Raised after bounded provider timeout retries are exhausted."""


class LLMRateLimitError(RouterError):
    """Raised after bounded provider rate-limit retries are exhausted."""


class LLMInvalidResponseError(RouterError):
    """Raised when structured-output correction is exhausted."""


class LLMProviderError(RouterError):
    """Raised for safe permanent or exhausted provider failures."""


class ActionDecision(BaseModel):
    """Strict schema for the first routing decision."""

    status: Literal["tool_call", "need_clarification"]
    tool_name: Literal[
        "predict_delay_risk",
        "explain_risk",
        "get_seller_history",
        "get_similar_past_orders",
    ] | None = None
    order_id: str | None = None
    seller_id: str | None = None
    clarification_question: str | None = None

    @model_validator(mode="after")
    def validate_status_fields(self) -> "ActionDecision":
        """Enforce mutually exclusive tool-call and clarification fields."""
        if self.status == "tool_call":
            if self.tool_name is None:
                raise ValueError("tool_call requires tool_name")
            if self.clarification_question is not None:
                raise ValueError("tool_call cannot include clarification_question")
        else:
            if (
                self.tool_name is not None
                or self.order_id is not None
                or self.seller_id is not None
            ):
                raise ValueError(
                    "need_clarification cannot include a tool or arguments"
                )
            if not self.clarification_question:
                raise ValueError(
                    "need_clarification requires clarification_question"
                )
        return self


class LoopDecision(BaseModel):
    """Strict schema for deciding whether to call again or answer."""

    status: Literal["tool_call", "final_answer", "need_clarification"]
    tool_name: Literal[
        "predict_delay_risk",
        "explain_risk",
        "get_seller_history",
        "get_similar_past_orders",
    ] | None = None
    order_id: str | None = None
    seller_id: str | None = None
    final_answer: str | None = None
    clarification_question: str | None = None

    @model_validator(mode="after")
    def validate_status_fields(self) -> "LoopDecision":
        """Require fields appropriate to exactly one loop action."""
        if self.status == "tool_call":
            if self.tool_name is None:
                raise ValueError("tool_call requires tool_name")
            if self.final_answer or self.clarification_question:
                raise ValueError("tool_call cannot include answer text")
        elif self.status == "final_answer":
            if not self.final_answer:
                raise ValueError("final_answer status requires final_answer")
        else:
            if not self.clarification_question:
                raise ValueError(
                    "need_clarification requires clarification_question"
                )
            if self.tool_name or self.order_id or self.seller_id:
                raise ValueError("clarification cannot include a tool call")
        return self


class FinalResponse(BaseModel):
    """Strict schema used when the tool-call cap forces completion."""

    answer: str


def _registry_by_name(registry: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate and index a registry by unique tool name."""
    indexed: dict[str, dict[str, Any]] = {}
    for tool in registry:
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Every registry entry requires a non-empty name.")
        if name in indexed:
            raise ValueError(f"Duplicate tool registry name: {name}")
        if not isinstance(tool.get("required_arguments"), dict):
            raise ValueError(f"Tool '{name}' requires an argument schema.")
        indexed[name] = tool
    return indexed


@lru_cache(maxsize=1)
def _get_gemini_client() -> genai.Client:
    """Load Gemini configuration without exposing the API key."""
    load_dotenv(override=False)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RouterError("GEMINI_API_KEY is not configured.")
    settings = get_settings()
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=int(settings.llm_timeout_seconds * 1_000),
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI:
    """Load OpenAI configuration without exposing the API key."""
    load_dotenv(override=False)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RouterError("OPENAI_API_KEY is not configured.")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    settings = get_settings()
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=0,
    )


def _provider_name() -> str:
    """Return the configured LLM provider."""
    load_dotenv(override=False)
    provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()
    if provider not in {"openai", "gemini"}:
        raise RouterError("LLM_PROVIDER must be either 'openai' or 'gemini'.")
    return provider


def _model_name() -> str:
    """Return the configured LLM model name."""
    load_dotenv(override=False)
    model = os.getenv("LLM_MODEL")
    if not model or model.startswith("replace-") or model.startswith("your-"):
        raise RouterError("LLM_MODEL is not configured with a real model name.")
    return model


def _temperature() -> float:
    """Use deterministic generation for repeatable router evaluation."""
    load_dotenv(override=False)
    return float(os.getenv("LLM_TEMPERATURE", "0"))


def _request_gemini_json(
    prompt: str,
    schema: type[BaseModel],
    max_output_tokens: int,
) -> str:
    """Request one JSON response from Gemini."""
    response = _get_gemini_client().models.generate_content(
        model=_model_name(),
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=_temperature(),
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    if not response.text:
        raise RouterError("Gemini returned an empty JSON response.")
    return response.text


def _request_openai_json(
    prompt: str,
    schema: type[BaseModel],
    max_output_tokens: int,
) -> str:
    """Request one JSON response from OpenAI and validate it locally later."""
    completion = _get_openai_client().chat.completions.create(
        model=_model_name(),
        temperature=_temperature(),
        max_tokens=max_output_tokens,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise JSON router. Return only JSON that "
                    "matches the schema requested in the user prompt."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": False,
            },
        },
    )
    content = completion.choices[0].message.content
    if not content:
        raise RouterError("OpenAI returned an empty JSON response.")
    return content


def _request_json(
    prompt: str,
    schema: type[BaseModel],
    max_output_tokens: int = 1_000,
) -> str:
    """Request a JSON response from the configured provider."""
    provider = _provider_name()
    request_function = (
        _request_openai_json if provider == "openai" else _request_gemini_json
    )
    settings = get_settings()
    for attempt in range(settings.llm_max_retries + 1):
        try:
            response, _ = measure_call(
                request_function,
                prompt,
                schema,
                max_output_tokens,
                event="llm_provider_call",
                provider_call_kind=schema.__name__,
                model_name=_model_name(),
                retry_count=attempt,
            )
            return response
        except Exception as exc:
            category, retryable = _classify_provider_exception(exc)
            if not retryable or attempt >= settings.llm_max_retries:
                exception_type = {
                    "llm_timeout": LLMTimeoutError,
                    "llm_rate_limit": LLMRateLimitError,
                }.get(category, LLMProviderError)
                raise exception_type(
                    f"{category}: provider request failed after "
                    f"{attempt + 1} attempt(s)."
                ) from exc

            delay_seconds = min(0.5 * (2**attempt), 4.0) + random.uniform(
                0.0, 0.25
            )
            emit_observation(
                "llm_retry_scheduled",
                level="warning",
                error_category=category,
                retry_count=attempt + 1,
                latency_ms=round(delay_seconds * 1_000, 3),
                provider_call_kind=schema.__name__,
                model_name=_model_name(),
            )
            time.sleep(delay_seconds)
    raise LLMProviderError("llm_provider_error: unreachable retry state.")


def _classify_provider_exception(exc: Exception) -> tuple[str, bool]:
    """Return safe category and whether a provider failure is transient."""
    if isinstance(exc, (APITimeoutError, TimeoutError)):
        return "llm_timeout", True
    if isinstance(exc, RateLimitError):
        return "llm_rate_limit", True
    if isinstance(exc, APIConnectionError):
        return "llm_provider_error", True

    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(exc, "code", None)
    if isinstance(exc, genai_errors.ClientError) and status_code == 429:
        return "llm_rate_limit", True
    if isinstance(exc, (APIStatusError, genai_errors.APIError)):
        if status_code == 429:
            return "llm_rate_limit", True
        if status_code in {408, 409, 500, 502, 503, 504}:
            return "llm_provider_error", True
        return "llm_provider_error", False
    if status_code == 429:
        return "llm_rate_limit", True
    if status_code in {408, 409, 500, 502, 503, 504}:
        return "llm_provider_error", True
    return "llm_provider_error", False


def _request_action(prompt: str) -> str:
    """Request one strict JSON routing decision."""
    return _request_json(prompt, ActionDecision, max_output_tokens=500)


def _validate_registry_arguments(
    decision: ActionDecision, registry: list[dict[str, Any]]
) -> dict[str, str]:
    """Return exactly the arguments declared for the chosen registry tool."""
    if decision.status == "need_clarification":
        return {}
    indexed = _registry_by_name(registry)
    if decision.tool_name not in indexed:
        raise ValueError(f"Tool is not present in registry: {decision.tool_name}")
    required = set(indexed[decision.tool_name]["required_arguments"])
    available_arguments = {
        "order_id": decision.order_id,
        "seller_id": decision.seller_id,
    }
    arguments = {
        name: value
        for name, value in available_arguments.items()
        if value is not None
    }
    supplied = set(arguments)
    if supplied != required:
        raise ValueError(
            f"Tool '{decision.tool_name}' requires exactly {sorted(required)}; "
            f"received {sorted(supplied)}."
        )
    empty = [name for name, value in arguments.items() if not value.strip()]
    if empty:
        raise ValueError(f"Arguments cannot be empty: {empty}")
    return arguments


def _parse_action_response(raw_response: str) -> ActionDecision:
    """Reject unknown JSON keys before Pydantic validation."""
    payload = json.loads(raw_response)
    if not isinstance(payload, dict):
        raise ValueError("Routing response must be a JSON object.")
    allowed_keys = {
        "status",
        "tool_name",
        "order_id",
        "seller_id",
        "clarification_question",
    }
    unknown_keys = set(payload).difference(allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown routing response keys: {sorted(unknown_keys)}")
    return ActionDecision.model_validate(payload)


def _parse_loop_response(raw_response: str) -> LoopDecision:
    """Reject unknown loop keys before validating the fixed schema."""
    payload = json.loads(raw_response)
    if not isinstance(payload, dict):
        raise ValueError("Loop response must be a JSON object.")
    allowed_keys = {
        "status",
        "tool_name",
        "order_id",
        "seller_id",
        "final_answer",
        "clarification_question",
    }
    unknown_keys = set(payload).difference(allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown loop response keys: {sorted(unknown_keys)}")
    return LoopDecision.model_validate(payload)


def decide_action(
    user_query: str, registry: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Choose one tool call or request clarification using strict JSON.

    Malformed JSON, schema failures, and registry/argument mismatches are
    rejected. The configured LLM receives one corrective retry; the router
    never guesses how to repair an invalid decision locally.
    """
    selected_registry = registry if registry is not None else TOOL_REGISTRY
    _registry_by_name(selected_registry)
    if not user_query.strip():
        return ActionDecision(
            status="need_clarification",
            clarification_question="What delivery-risk question can I help with?",
        ).model_dump()

    base_prompt = f"""You route delivery-risk questions to deterministic tools.

Available tool registry:
{json.dumps(selected_registry, indent=2)}

Routing rules:
- Choose exactly one tool only when its required identifier is explicitly
  present in the user's query.
- Copy identifiers exactly; never invent, shorten, or alter them.
- Risk/probability/will-it-be-late questions use predict_delay_risk.
- Why/explanation/feature-contribution questions use explain_risk.
- Seller performance/history/reliability questions use get_seller_history.
- Similar/comparable/past-order questions use get_similar_past_orders.
- If the required order_id or seller_id is absent or ambiguous, return
  need_clarification and ask for that identifier.
- If multiple order IDs or seller IDs appear and the user does not clearly
  specify which one to use, return need_clarification. Do not guess.
- Phrases like "my order", "this order", "the order", or "it" are not valid
  identifiers. If no exact order_id is written in the query, return
  need_clarification.
- Never return status "tool_call" unless you also provide the exact required
  order_id or seller_id field and leave clarification_question null.
- Return only JSON matching the supplied schema.

User query:
{user_query}
"""

    correction = ""
    last_error: Exception | None = None
    # One initial attempt plus exactly one retry for malformed/invalid output.
    for attempt in range(2):
        raw_response = _request_action(base_prompt + correction)
        try:
            decision = _parse_action_response(raw_response)
            arguments = _validate_registry_arguments(decision, selected_registry)
            return {
                "status": decision.status,
                "tool_name": decision.tool_name,
                "arguments": arguments,
                "clarification_question": decision.clarification_question,
            }
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            correction = (
                "\nYour previous response was rejected because it violated the "
                f"schema or registry: {exc}. Return a corrected JSON decision only."
            )
            if attempt == 1:
                break
    raise LLMInvalidResponseError(
        "The LLM failed to return a valid routing decision after one retry: "
        f"{last_error}"
    )


def _decide_next_step(
    user_query: str,
    trace: list[dict[str, Any]],
    registry: list[dict[str, Any]],
) -> dict[str, Any]:
    """Decide whether actual tool results are sufficient for an answer."""
    prompt = f"""You are the control step in a delivery-risk tool loop.

Original user query:
{user_query}

Available tool registry:
{json.dumps(registry, indent=2)}

Trace containing decisions and actual tool outputs:
{json.dumps(trace, indent=2)}

Rules:
- Return final_answer when existing tool outputs answer the original request.
- If the user asked for both risk and a reason, and prediction is high risk,
  call explain_risk with the exact same order_id before answering.
- Call another tool only when it is necessary for the original request.
- Never repeat an identical tool call.
- Use only identifiers present in the original query or tool results.
- A final answer may reference only values present in tool outputs. Do not add
  outside facts, assumptions, causes, or invented values.
- If a required identifier is unavailable, return need_clarification.
- Return only JSON matching the supplied schema.
"""
    correction = ""
    last_error: Exception | None = None
    for attempt in range(2):
        raw_response = _request_json(
            prompt + correction,
            LoopDecision,
            max_output_tokens=1_000,
        )
        try:
            decision = _parse_loop_response(raw_response)
            action = ActionDecision(
                status=(
                    "tool_call"
                    if decision.status == "tool_call"
                    else "need_clarification"
                ),
                tool_name=decision.tool_name,
                order_id=decision.order_id,
                seller_id=decision.seller_id,
                clarification_question=decision.clarification_question,
            ) if decision.status != "final_answer" else None
            arguments = (
                _validate_registry_arguments(action, registry) if action else {}
            )
            return {
                "status": decision.status,
                "tool_name": decision.tool_name,
                "arguments": arguments,
                "final_answer": decision.final_answer,
                "clarification_question": decision.clarification_question,
            }
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            correction = (
                "\nThe previous response was invalid: "
                f"{exc}. Return corrected JSON only."
            )
            if attempt == 1:
                break
    raise LLMInvalidResponseError(
        f"The LLM failed to return a valid loop decision after one retry: {last_error}"
    )


def _grounded_final_answer(
    user_query: str, trace: list[dict[str, Any]]
) -> str:
    """Generate a final answer constrained to recorded tool outputs."""
    prompt = f"""Answer the user's delivery-risk question using only the actual
tool results in the trace below. Do not add outside facts, infer causes, or
invent values. If the tools lack something, say it is unavailable. Keep the
answer concise and preserve the tools' correlational-not-causal caveat when an
explanation is present.

User query:
{user_query}

Trace:
{json.dumps(trace, indent=2)}

Return only JSON matching the supplied schema.
"""
    last_error: Exception | None = None
    correction = ""
    for attempt in range(2):
        raw_response = _request_json(
            prompt + correction,
            FinalResponse,
            max_output_tokens=1_000,
        )
        try:
            payload = json.loads(raw_response)
            if not isinstance(payload, dict) or set(payload) != {"answer"}:
                raise ValueError("Final response must contain only 'answer'.")
            return FinalResponse.model_validate(payload).answer
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            correction = f"\nPrevious response invalid: {exc}. Return JSON only."
            if attempt == 1:
                break
    raise LLMInvalidResponseError(
        f"The LLM failed to return a grounded final answer after one retry: {last_error}"
    )


def run_agent(
    user_query: str, registry: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Route, execute tools, optionally loop back, and return a traced answer."""
    selected_registry = registry if registry is not None else TOOL_REGISTRY
    indexed_registry = _registry_by_name(selected_registry)
    initial_decision = decide_action(user_query, selected_registry)
    trace: list[dict[str, Any]] = [
        {"event": "decision", "decision": initial_decision}
    ]

    if initial_decision["status"] == "need_clarification":
        return {
            "ok": True,
            "status": "need_clarification",
            "answer": initial_decision["clarification_question"],
            "tool_call_count": 0,
            "trace": trace,
        }

    load_dotenv(override=False)
    max_tool_calls = int(os.getenv("LLM_MAX_TOOL_CALLS", "3"))
    if max_tool_calls < 1:
        raise RouterError("LLM_MAX_TOOL_CALLS must be at least 1.")

    next_action = initial_decision
    tool_call_count = 0
    seen_calls: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

    while True:
        tool_name = next_action["tool_name"]
        arguments = next_action["arguments"]
        if tool_name not in indexed_registry or tool_name not in TOOL_FUNCTIONS:
            raise RouterError(f"Tool is registered but not executable: {tool_name}")
        call_signature = (tool_name, tuple(sorted(arguments.items())))
        if call_signature in seen_calls:
            answer = _grounded_final_answer(user_query, trace)
            return {
                "ok": True,
                "status": "completed",
                "answer": answer,
                "tool_call_count": tool_call_count,
                "stop_reason": "duplicate_call_prevented",
                "trace": trace,
            }
        seen_calls.add(call_signature)

        started = perf_counter()
        result = TOOL_FUNCTIONS[tool_name](**arguments)
        elapsed_ms = (perf_counter() - started) * 1_000
        emit_observation(
            "deterministic_tool_call",
            selected_tool=tool_name,
            latency_ms=round(elapsed_ms, 3),
            outcome_ok=bool(result.get("ok", False)),
            agent_implementation="plain_python",
        )
        tool_call_count += 1
        trace.append(
            {
                "event": "tool_result",
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "elapsed_ms": round(elapsed_ms, 3),
            }
        )

        # Hard cap prevents an incorrect model decision from creating an
        # infinite tool loop or unbounded API/tool costs.
        if tool_call_count >= max_tool_calls:
            answer = _grounded_final_answer(user_query, trace)
            return {
                "ok": True,
                "status": "completed",
                "answer": answer,
                "tool_call_count": tool_call_count,
                "stop_reason": "max_tool_calls_reached",
                "trace": trace,
            }

        loop_decision = _decide_next_step(
            user_query, trace, selected_registry
        )
        trace.append({"event": "decision", "decision": loop_decision})
        if loop_decision["status"] == "final_answer":
            return {
                "ok": True,
                "status": "completed",
                "answer": loop_decision["final_answer"],
                "tool_call_count": tool_call_count,
                "stop_reason": "enough_information",
                "trace": trace,
            }
        if loop_decision["status"] == "need_clarification":
            return {
                "ok": True,
                "status": "need_clarification",
                "answer": loop_decision["clarification_question"],
                "tool_call_count": tool_call_count,
                "trace": trace,
            }
        next_action = loop_decision
