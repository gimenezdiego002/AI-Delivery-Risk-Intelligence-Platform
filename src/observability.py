"""Framework-neutral timing hooks shared by API, agents, and tools."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Callable, Iterator, TypeVar


_T = TypeVar("_T")
_LOGGER = logging.getLogger("delivery_risk")


def emit_observation(
    event: str,
    *,
    level: str = "info",
    **metadata: Any,
) -> None:
    """Emit safe metadata through the configured API logger, if present."""
    getattr(_LOGGER, level)(
        event,
        extra={
            "event_name": event,
            "event_metadata": metadata,
            "app_env": os.getenv("APP_ENV", "development"),
        },
    )


@contextmanager
def timed_operation(event: str, **metadata: Any) -> Iterator[None]:
    """Log monotonic duration and outcome for one bounded operation."""
    started = perf_counter()
    try:
        yield
    except Exception:
        emit_observation(
            event,
            level="error",
            **metadata,
            latency_ms=round((perf_counter() - started) * 1_000, 3),
            outcome_ok=False,
        )
        raise
    else:
        emit_observation(
            event,
            **metadata,
            latency_ms=round((perf_counter() - started) * 1_000, 3),
            outcome_ok=True,
        )


def measure_call(
    function: Callable[..., _T],
    *args: Any,
    event: str,
    **metadata: Any,
) -> tuple[_T, float]:
    """Call a function, log its duration, and return result plus milliseconds."""
    started = perf_counter()
    try:
        result = function(*args)
    except Exception:
        elapsed_ms = (perf_counter() - started) * 1_000
        emit_observation(
            event,
            level="error",
            **metadata,
            latency_ms=round(elapsed_ms, 3),
            outcome_ok=False,
        )
        raise
    elapsed_ms = (perf_counter() - started) * 1_000
    emit_observation(
        event,
        **metadata,
        latency_ms=round(elapsed_ms, 3),
        outcome_ok=True,
    )
    return result, elapsed_ms
