"""Mocked tests for bounded provider timeout and retry behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.agent.router import (
    ActionDecision,
    LLMProviderError,
    LLMTimeoutError,
    _request_json,
)


def _settings(retries: int = 2) -> SimpleNamespace:
    return SimpleNamespace(llm_max_retries=retries)


def _request_with(side_effect, retries: int = 2):
    request = Mock(side_effect=side_effect)
    patches = (
        patch("src.agent.router._provider_name", return_value="openai"),
        patch("src.agent.router._model_name", return_value="test-model"),
        patch("src.agent.router._request_openai_json", request),
        patch("src.agent.router.get_settings", return_value=_settings(retries)),
        patch("src.agent.router.time.sleep"),
        patch("src.agent.router.random.uniform", return_value=0.0),
    )
    return request, patches


def test_timeout_retries_are_bounded() -> None:
    request, patches = _request_with(TimeoutError("provider timeout"))
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4] as sleep,
        patches[5],
        pytest.raises(LLMTimeoutError),
    ):
        _request_json("prompt", ActionDecision)
    assert request.call_count == 3
    assert sleep.call_count == 2


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_transient_status_retries_then_succeeds(status_code: int) -> None:
    transient = RuntimeError("transient")
    transient.status_code = status_code  # type: ignore[attr-defined]
    request, patches = _request_with([transient, "{}"])
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
    ):
        assert _request_json("prompt", ActionDecision) == "{}"
    assert request.call_count == 2


def test_permanent_client_error_is_not_retried() -> None:
    permanent = RuntimeError("bad request")
    permanent.status_code = 400  # type: ignore[attr-defined]
    request, patches = _request_with(permanent)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4] as sleep,
        patches[5],
        pytest.raises(LLMProviderError),
    ):
        _request_json("prompt", ActionDecision)
    assert request.call_count == 1
    sleep.assert_not_called()
