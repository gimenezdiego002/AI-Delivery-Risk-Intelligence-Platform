"""Measure plain-Python and LangGraph routing/full-agent latency."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.agent.langgraph_agent import run_langgraph_agent
from src.agent.router import RouterError, decide_action, run_agent


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUTING_CASES_PATH = PROJECT_ROOT / "tests" / "agent_eval_queries.csv"
OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "phase_9_agent_comparison_metrics.json"
)
LANGGRAPH_ROUTING_REPORT = (
    PROJECT_ROOT / "reports" / "phase_9_langgraph_router_evaluation.json"
)
PLAIN_ROUTING_CHECKPOINT = (
    PROJECT_ROOT / "reports" / "phase_9_plain_router_latency_checkpoint.json"
)
FULL_AGENT_CASES = [
    {
        "name": "direct_prediction",
        "query": (
            "Predict late-delivery risk for order "
            "be55f985440dddd650b389a55db8e49c."
        ),
    },
    {
        "name": "conditional_explanation",
        "query": (
            "Predict risk for order be55f985440dddd650b389a55db8e49c "
            "and if it is high, explain why."
        ),
    },
    {
        "name": "clarification",
        "query": "Can you check whether order 123 or order 456 will be late?",
    },
]


def _latency_summary(values: list[float]) -> dict[str, float]:
    """Return average and median measurements in milliseconds."""
    return {
        "average_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
    }


def _actual_action(decision: dict[str, Any]) -> str:
    """Normalize either implementation's first-decision response."""
    return (
        decision.get("tool_name")
        if decision.get("status") == "tool_call"
        else str(decision.get("status"))
    )


def _measure_routing(
    label: str,
    decide: Callable[[str], dict[str, Any]],
    cases: pd.DataFrame,
) -> dict[str, Any]:
    """Measure every labeled first action for one implementation."""
    latencies: list[float] = []
    rows: list[dict[str, Any]] = []
    for index, row in cases.iterrows():
        started = time.perf_counter()
        decision = decide(row["query"])
        elapsed_ms = (time.perf_counter() - started) * 1_000
        actual = _actual_action(decision)
        latencies.append(elapsed_ms)
        rows.append(
            {
                "query": row["query"],
                "expected": row["expected_tool"],
                "actual": actual,
                "correct": actual == row["expected_tool"],
                "latency_ms": elapsed_ms,
            }
        )
        print(
            f"[{label} route {index + 1:02d}/{len(cases)}] "
            f"{actual} {elapsed_ms:.2f}ms"
        )
    return {
        "correct": sum(item["correct"] for item in rows),
        "total": len(rows),
        "accuracy": statistics.fmean(
            float(item["correct"]) for item in rows
        ),
        "latency": _latency_summary(latencies),
        "incorrect_cases": [item for item in rows if not item["correct"]],
    }


def _measure_full_agent(
    label: str,
    run: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """Measure representative full-agent paths for one implementation."""
    rows: list[dict[str, Any]] = []
    for case in FULL_AGENT_CASES:
        started = time.perf_counter()
        last_error: str | None = None
        result: dict[str, Any] | None = None
        attempts = 0
        for attempts in range(1, 5):
            try:
                candidate = run(case["query"])
                if candidate.get("status") != "error":
                    result = candidate
                    break
                last_error = str(candidate.get("error"))
            except RouterError as exc:
                last_error = str(exc)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        if result is None:
            result = {
                "status": "error",
                "tool_call_count": 0,
                "error": last_error,
            }
        rows.append(
            {
                "name": case["name"],
                "latency_ms": elapsed_ms,
                "status": result.get("status"),
                "tool_call_count": result.get("tool_call_count"),
                "attempts": attempts,
                "error": result.get("error"),
            }
        )
        print(
            f"[{label} full] {case['name']}: "
            f"{elapsed_ms:.2f}ms, status={result.get('status')}"
        )
    return {
        "latency": _latency_summary(
            [item["latency_ms"] for item in rows]
        ),
        "cases": rows,
    }


def main() -> None:
    """Run the comparison and save all measured values."""
    cases = pd.read_csv(ROUTING_CASES_PATH)
    langgraph_report = json.loads(
        LANGGRAPH_ROUTING_REPORT.read_text(encoding="utf-8")
    )
    plain_checkpoint = json.loads(
        PLAIN_ROUTING_CHECKPOINT.read_text(encoding="utf-8")
    )
    started = time.perf_counter()
    result = {
        "routing": {
            "plain_python": _measure_routing(
                "plain_python", decide_action, cases
            ) if not PLAIN_ROUTING_CHECKPOINT.exists() else {
                "correct": plain_checkpoint["correct_queries"],
                "total": plain_checkpoint["measured_queries"],
                "accuracy": plain_checkpoint["accuracy"],
                "latency": {
                    "average_ms": plain_checkpoint["average_latency_ms"],
                    "median_ms": plain_checkpoint["median_latency_ms"],
                    "minimum_ms": plain_checkpoint["minimum_latency_ms"],
                    "maximum_ms": plain_checkpoint["maximum_latency_ms"],
                },
                "incorrect_cases": [],
                "source": str(
                    PLAIN_ROUTING_CHECKPOINT.relative_to(PROJECT_ROOT)
                ),
            },
            "langgraph": {
                "correct": langgraph_report["correct_queries"],
                "total": langgraph_report["total_queries"],
                "accuracy": langgraph_report["tool_selection_accuracy"],
                "latency": {
                    "average_ms": langgraph_report["average_latency_ms"],
                    "median_ms": langgraph_report["median_latency_ms"],
                },
                "incorrect_cases": langgraph_report["incorrect_cases"],
                "source": str(LANGGRAPH_ROUTING_REPORT.relative_to(PROJECT_ROOT)),
            },
        },
        "full_agent": {
            "plain_python": _measure_full_agent(
                "plain_python", run_agent
            ),
            "langgraph": _measure_full_agent(
                "langgraph", run_langgraph_agent
            ),
        },
    }
    result["total_measurement_seconds"] = time.perf_counter() - started
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved measured comparison to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
