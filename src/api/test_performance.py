"""Measure local FastAPI endpoint latency for Phase 6.

Run the API first:
    uvicorn src.api.main:app --reload

Then, in a second terminal:
    python -m src.api.test_performance

This script is intentionally not a pytest test. It makes real HTTP requests,
and the `/agent/query` endpoint can make real LLM API calls that may incur
provider cost.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_COUNT = 10

ORDER_ID = "be55f985440dddd650b389a55db8e49c"
SELLER_ID = "3078096983cf766a32a06257648502d1"


@dataclass(frozen=True)
class EndpointCase:
    """One HTTP endpoint to measure."""

    name: str
    method: str
    path: str
    body: dict[str, Any] | None = None


ENDPOINTS = [
    EndpointCase("health", "GET", "/health"),
    EndpointCase("risk", "GET", f"/orders/{ORDER_ID}/risk"),
    EndpointCase("explanation", "GET", f"/orders/{ORDER_ID}/explanation"),
    EndpointCase("seller_history", "GET", f"/sellers/{SELLER_ID}/history"),
    EndpointCase(
        "agent_query",
        "POST",
        "/agent/query",
        {
            "query": (
                f"First predict delay risk for order {ORDER_ID} and if it is "
                "high, explain why."
            )
        },
    ),
]


def _request(case: EndpointCase) -> tuple[int, dict[str, Any]]:
    """Make one HTTP request and return status code plus parsed JSON."""
    data = None
    headers = {"Accept": "application/json"}
    if case.body is not None:
        data = json.dumps(case.body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        f"{BASE_URL}{case.path}",
        data=data,
        headers=headers,
        method=case.method,
    )
    try:
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status, payload
    except HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        return exc.code, payload


def _p95(milliseconds: list[float]) -> float:
    """Return nearest-rank p95, which is stable for small sample sizes."""
    ordered = sorted(milliseconds)
    index = max(0, int(0.95 * len(ordered) + 0.999999) - 1)
    return ordered[index]


def measure(case: EndpointCase) -> dict[str, float]:
    """Measure one endpoint multiple times and return latency statistics."""
    latencies_ms: list[float] = []
    for _ in range(REQUEST_COUNT):
        start = time.perf_counter()
        status_code, payload = _request(case)
        elapsed_ms = (time.perf_counter() - start) * 1_000
        if status_code >= 400:
            raise RuntimeError(
                f"{case.name} returned HTTP {status_code}: {payload}"
            )
        latencies_ms.append(elapsed_ms)

    return {
        "average_ms": statistics.fmean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "p95_ms": _p95(latencies_ms),
    }


def main() -> None:
    """Print average, median, and p95 latency for every Phase 6 endpoint."""
    print(f"Measuring {REQUEST_COUNT} requests per endpoint at {BASE_URL}\n")
    print("| Endpoint | Average ms | Median ms | P95 ms |")
    print("|---|---:|---:|---:|")
    try:
        for case in ENDPOINTS:
            metrics = measure(case)
            print(
                f"| {case.name} | {metrics['average_ms']:.2f} | "
                f"{metrics['median_ms']:.2f} | {metrics['p95_ms']:.2f} |"
            )
    except URLError as exc:
        raise SystemExit(
            "Could not reach the API. Start it first with: "
            "uvicorn src.api.main:app --reload"
        ) from exc


if __name__ == "__main__":
    main()
