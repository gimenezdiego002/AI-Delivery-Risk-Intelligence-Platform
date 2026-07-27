"""Measure Phase 10 HTTP latency without hiding request failures.

Run the API separately, then set ``BENCHMARK_ENVIRONMENT`` (for example
``local`` or ``docker_local``) and execute:

    python -m src.api.benchmark_endpoints

Agent cases make paid provider calls. ``BENCHMARK_REQUEST_COUNT`` defaults to
three to limit accidental cost. Authentication uses ``API_KEY`` only as an
outbound header and never writes it to the report.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
ENVIRONMENT = os.getenv("BENCHMARK_ENVIRONMENT", "local")
REQUEST_COUNT = int(os.getenv("BENCHMARK_REQUEST_COUNT", "3"))
TIMEOUT_SECONDS = float(os.getenv("BENCHMARK_TIMEOUT_SECONDS", "120"))
OUTPUT_PATH = Path(
    os.getenv(
        "PHASE10_PERFORMANCE_OUTPUT",
        PROJECT_ROOT
        / "reports"
        / f"phase_10_performance_{ENVIRONMENT}.json",
    )
)
ORDER_ID = "be55f985440dddd650b389a55db8e49c"
SELLER_ID = "3078096983cf766a32a06257648502d1"


@dataclass(frozen=True)
class BenchmarkCase:
    """One repeatable API request."""

    name: str
    method: str
    path: str
    body: dict[str, Any] | None = None


CASES = (
    BenchmarkCase("health", "GET", "/health"),
    BenchmarkCase("prediction", "GET", f"/orders/{ORDER_ID}/risk"),
    BenchmarkCase(
        "explanation", "GET", f"/orders/{ORDER_ID}/explanation"
    ),
    BenchmarkCase(
        "seller_history", "GET", f"/sellers/{SELLER_ID}/history"
    ),
    BenchmarkCase(
        "plain_agent",
        "POST",
        "/agent/query",
        {"query": f"Predict and explain order {ORDER_ID}."},
    ),
    BenchmarkCase(
        "langgraph_direct",
        "POST",
        "/agent/langgraph/query",
        {"query": f"Predict risk for order {ORDER_ID}."},
    ),
    BenchmarkCase(
        "langgraph_two_tool",
        "POST",
        "/agent/langgraph/query",
        {"query": f"Predict and explain order {ORDER_ID}."},
    ),
)


def _p95(values: list[float]) -> float:
    """Return nearest-rank p95, including for deliberately small samples."""
    ordered = sorted(values)
    index = max(0, int(0.95 * len(ordered) + 0.999999) - 1)
    return ordered[index]


def _send(case: BenchmarkCase) -> tuple[int, dict[str, Any]]:
    body = json.dumps(case.body).encode() if case.body is not None else None
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    api_key = os.getenv("API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    request = Request(
        f"{BASE_URL}{case.path}",
        data=body,
        headers=headers,
        method=case.method,
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, json.loads(response.read().decode())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _measure(case: BenchmarkCase) -> dict[str, Any]:
    latencies: list[float] = []
    errors: list[dict[str, Any]] = []
    for attempt in range(REQUEST_COUNT):
        started = perf_counter()
        try:
            status, payload = _send(case)
        except (URLError, TimeoutError) as exc:
            status = 0
            payload = {"error": {"code": type(exc).__name__}}
        elapsed_ms = (perf_counter() - started) * 1_000
        latencies.append(elapsed_ms)
        if status >= 400 or status == 0 or payload.get("status") == "error":
            errors.append(
                {
                    "attempt": attempt + 1,
                    "status_code": status,
                    "error_code": payload.get("error", {}).get("code"),
                }
            )
    return {
        "count": REQUEST_COUNT,
        "average_ms": statistics.fmean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": _p95(latencies),
        "minimum_ms": min(latencies),
        "maximum_ms": max(latencies),
        "error_count": len(errors),
        "error_rate": len(errors) / REQUEST_COUNT,
        "errors": errors,
    }


def main() -> None:
    """Measure all cases and persist metadata without secrets or payloads."""
    if REQUEST_COUNT < 1:
        raise ValueError("BENCHMARK_REQUEST_COUNT must be at least 1.")
    results = {
        "environment": ENVIRONMENT,
        "base_url": BASE_URL,
        "request_count_per_case": REQUEST_COUNT,
        "client_retries": 0,
        "cases": {case.name: _measure(case) for case in CASES},
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
