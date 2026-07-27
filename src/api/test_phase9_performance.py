"""Measure Docker/local latency for the Phase 9 agent comparison endpoints.

Start the API first, then run:

    python -m src.api.test_phase9_performance

Set ``API_BASE_URL`` to change the target and ``PHASE9_REQUEST_COUNT`` to
control paid LLM calls. The default is three requests per case.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "reports" / "phase_9_docker_performance.json"
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_COUNT = int(os.getenv("PHASE9_REQUEST_COUNT", "3"))
ORDER_ID = "be55f985440dddd650b389a55db8e49c"


CASES = [
    {
        "name": "direct_prediction",
        "method": "GET",
        "path": f"/orders/{ORDER_ID}/risk",
        "body": None,
    },
    {
        "name": "plain_agent_two_tool",
        "method": "POST",
        "path": "/agent/query",
        "body": {
            "query": (
                f"First predict delay risk for order {ORDER_ID} and if it is "
                "high, explain why."
            )
        },
    },
    {
        "name": "langgraph_direct",
        "method": "POST",
        "path": "/agent/langgraph/query",
        "body": {"query": f"Predict risk for order {ORDER_ID}."},
    },
    {
        "name": "langgraph_two_tool",
        "method": "POST",
        "path": "/agent/langgraph/query",
        "body": {
            "query": (
                f"Predict risk for order {ORDER_ID} and explain its risk."
            )
        },
    },
]


def _request(case: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Send one JSON request and return status plus parsed response."""
    data = (
        json.dumps(case["body"]).encode("utf-8")
        if case["body"] is not None
        else None
    )
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(
        f"{BASE_URL}{case['path']}",
        data=data,
        headers=headers,
        method=case["method"],
    )
    try:
        with urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _measure(case: dict[str, Any]) -> dict[str, Any]:
    """Measure one endpoint and retain only safe response metadata."""
    latencies: list[float] = []
    last_payload: dict[str, Any] = {}
    retry_count = 0
    for _ in range(REQUEST_COUNT):
        started = time.perf_counter()
        for attempt in range(3):
            status, last_payload = _request(case)
            response_failed = status >= 400 or last_payload.get("status") == "error"
            if not response_failed:
                break
            if attempt < 2:
                retry_count += 1
        latencies.append((time.perf_counter() - started) * 1_000)
        if status >= 400 or last_payload.get("status") == "error":
            raise RuntimeError(
                f"{case['name']} returned HTTP {status}: {last_payload}"
            )
    return {
        "request_count": REQUEST_COUNT,
        "average_ms": statistics.fmean(latencies),
        "median_ms": statistics.median(latencies),
        "minimum_ms": min(latencies),
        "maximum_ms": max(latencies),
        "client_retry_count": retry_count,
        "last_status": last_payload.get("status", "success"),
        "last_tool_call_count": last_payload.get("tool_call_count"),
    }


def main() -> None:
    """Run every measurement and save the real results."""
    if REQUEST_COUNT < 1:
        raise ValueError("PHASE9_REQUEST_COUNT must be at least 1.")
    result = {
        "base_url": BASE_URL,
        "request_count_per_case": REQUEST_COUNT,
        "cases": {case["name"]: _measure(case) for case in CASES},
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
