"""Evaluate complete LangGraph workflows separately from first-tool routing."""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from src.agent.langgraph_agent import run_langgraph_agent


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PATH = (
    PROJECT_ROOT / "tests" / "langgraph_multistep_eval_queries.json"
)
RESULTS_PATH = (
    PROJECT_ROOT / "reports" / "phase_9_langgraph_multistep_evaluation.json"
)
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(-?\d+(?:\.\d+)?%?)(?![A-Za-z0-9])"
)


def _numeric_values(value: Any) -> list[float]:
    """Collect public numerical values from deterministic tool results."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [
            item
            for nested in value.values()
            for item in _numeric_values(nested)
        ]
    if isinstance(value, list):
        return [item for nested in value for item in _numeric_values(nested)]
    if isinstance(value, str):
        return [
            abs(float(token.rstrip("%")))
            for token in NUMBER_PATTERN.findall(value.replace("T", " "))
        ]
    return []


def _answer_is_numerically_grounded(
    answer: str, tool_results: list[dict[str, Any]]
) -> bool:
    """Reject numerical claims that cannot be traced to deterministic results."""
    source_values = _numeric_values(tool_results)
    for token in NUMBER_PATTERN.findall(answer):
        is_percent = token.endswith("%")
        numeric = float(token.rstrip("%"))
        if numeric in {1.0, 2.0, 3.0, 4.0, 5.0}:
            # Allow ordinary list numbering in prose.
            continue
        candidates = [numeric / 100] if is_percent else [numeric]
        if not is_percent and numeric > 1:
            candidates.append(numeric / 100)
        if not any(
            abs(candidate - source) <= max(0.005, abs(source) * 0.01)
            for candidate in candidates
            for source in source_values
        ):
            return False
    return True


def evaluate_multistep() -> dict[str, Any]:
    """Run every full workflow and save sequence, status, grounding, and cap checks."""
    cases = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    total_started = time.perf_counter()

    for index, case in enumerate(cases, start=1):
        print(f"[{index:02d}/{len(cases)}] {case['name']}")
        started = time.perf_counter()
        result = run_langgraph_agent(case["query"])
        elapsed_ms = (time.perf_counter() - started) * 1_000
        actual_tools = [
            item["tool_name"] for item in result.get("tool_results", [])
        ]
        grounded = (
            _answer_is_numerically_grounded(
                result.get("answer", ""),
                result.get("tool_results", []),
            )
            if case["grounding_required"]
            else True
        )
        cap_respected = (
            result.get("tool_call_count", 0)
            <= case["maximum_allowed_tool_calls"]
        )
        sequence_correct = actual_tools == case["expected_tools"]
        status_correct = result.get("status") == case["expected_status"]
        actual_first_action = (
            actual_tools[0]
            if actual_tools
            else result.get("status")
        )
        first_action_correct = (
            actual_first_action == case["expected_first_action"]
        )
        clarification_occurred = (
            result.get("status") == "need_clarification"
        )
        clarification_correct = (
            clarification_occurred == case["clarification_expected"]
        )
        stop_reason_correct = (
            result.get("stop_reason") == case["expected_stop_reason"]
            if "expected_stop_reason" in case
            else True
        )
        row = {
            "name": case["name"],
            "query": case["query"],
            "expected_tool_sequence": case["expected_tools"],
            "actual_tool_sequence": actual_tools,
            "expected_first_action": case["expected_first_action"],
            "actual_first_action": actual_first_action,
            "expected_final_status": case["expected_status"],
            "actual_final_status": result.get("status"),
            "first_action_correct": first_action_correct,
            "sequence_correct": sequence_correct,
            "status_correct": status_correct,
            "grounded": grounded,
            "clarification_expected": case["clarification_expected"],
            "clarification_occurred": clarification_occurred,
            "clarification_correct": clarification_correct,
            "maximum_allowed_tool_calls": case[
                "maximum_allowed_tool_calls"
            ],
            "tool_call_cap_respected": cap_respected,
            "stop_reason": result.get("stop_reason"),
            "stop_reason_correct": stop_reason_correct,
            "tool_call_count": result.get("tool_call_count"),
            "latency_ms": elapsed_ms,
            "answer": result.get("answer"),
            "error": result.get("error"),
            "tool_results": result.get("tool_results"),
            "trace": result.get("trace"),
            "passed": (
                first_action_correct
                and sequence_correct
                and status_correct
                and grounded
                and clarification_correct
                and cap_respected
                and stop_reason_correct
            ),
        }
        rows.append(row)
        print(
            f"    expected={case['expected_tools']} actual={actual_tools} "
            f"status={result.get('status')} grounded={grounded} "
            f"latency={elapsed_ms:.2f}ms"
        )

    latencies = [row["latency_ms"] for row in rows]
    return {
        "implementation": "langgraph",
        "total_cases": len(rows),
        "passed_cases": sum(row["passed"] for row in rows),
        "all_passed": all(row["passed"] for row in rows),
        "average_latency_ms": statistics.fmean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "total_evaluation_seconds": time.perf_counter() - total_started,
        "cases": rows,
    }


def main() -> None:
    """Run and persist the measured full-workflow evaluation."""
    result = evaluate_multistep()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nLANGGRAPH MULTI-STEP EVALUATION")
    print(f"Passed: {result['passed_cases']}/{result['total_cases']}")
    print(f"Average latency: {result['average_latency_ms']:.2f}ms")
    print(f"Median latency: {result['median_latency_ms']:.2f}ms")


if __name__ == "__main__":
    main()
