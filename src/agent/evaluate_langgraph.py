"""Measure LangGraph first-decision routing on the Phase 5 labeled dataset."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from google.genai import errors as genai_errors
from openai import APIError, RateLimitError

from src.agent.langgraph_agent import decide_langgraph_action
from src.agent.router import RouterError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PATH = PROJECT_ROOT / "tests" / "agent_eval_queries.csv"
RESULTS_PATH = (
    PROJECT_ROOT / "reports" / "phase_9_langgraph_router_evaluation.json"
)
DETAILS_PATH = (
    PROJECT_ROOT / "reports" / "phase_9_langgraph_router_predictions.csv"
)


def _decide_with_provider_retry(query: str) -> dict[str, Any]:
    """Retry transient provider failures without scoring them as model choices."""
    for attempt in range(5):
        try:
            return decide_langgraph_action(query)
        except (genai_errors.ClientError, genai_errors.ServerError) as exc:
            status_code = getattr(exc, "code", getattr(exc, "status_code", None))
            retryable = status_code in {429, 500, 502, 503, 504}
            if not retryable or attempt == 4:
                raise
        except (RateLimitError, APIError) as exc:
            retryable = getattr(exc, "status_code", None) in {
                429,
                500,
                502,
                503,
                504,
            }
            if not retryable or attempt == 4:
                raise
        wait_seconds = 2 ** attempt
        print(f"    Provider temporarily unavailable; retrying in {wait_seconds}s")
        time.sleep(wait_seconds)
    raise RuntimeError("Unreachable provider retry state.")


def evaluate_langgraph_router(
    evaluation_path: Path = EVALUATION_PATH,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run all labeled queries and return summary plus row-level measurements."""
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Evaluation set not found: {evaluation_path}")

    load_dotenv(override=False)
    evaluation = pd.read_csv(evaluation_path)
    required_columns = {"query", "expected_tool"}
    if not required_columns.issubset(evaluation.columns):
        raise ValueError(
            f"Evaluation CSV must contain {sorted(required_columns)}."
        )

    rows: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for index, item in evaluation.iterrows():
        print(f"[{index + 1:02d}/{len(evaluation)}] {item['query']}")
        started = time.perf_counter()
        error_message = None
        try:
            decision = _decide_with_provider_retry(item["query"])
            actual_action = (
                decision["tool_name"]
                if decision["status"] == "tool_call"
                else decision["status"]
            )
        except RouterError as exc:
            actual_action = "router_error"
            error_message = str(exc)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        correct = actual_action == item["expected_tool"]
        rows.append(
            {
                "query": item["query"],
                "expected_action": item["expected_tool"],
                "actual_action": actual_action,
                "correct": correct,
                "latency_ms": elapsed_ms,
                "error": error_message,
            }
        )
        print(
            f"    expected={item['expected_tool']} actual={actual_action} "
            f"correct={correct} latency={elapsed_ms:.2f}ms"
        )

    details = pd.DataFrame(rows)
    latencies = details["latency_ms"].tolist()
    per_tool: dict[str, dict[str, Any]] = {}
    for expected, group in details.groupby("expected_action", sort=True):
        correct_count = int(group["correct"].sum())
        per_tool[expected] = {
            "correct": correct_count,
            "total": len(group),
            "accuracy": correct_count / len(group),
            "average_latency_ms": float(group["latency_ms"].mean()),
        }

    mistakes = details.loc[~details["correct"]]
    summary = {
        "implementation": "langgraph",
        "total_queries": len(details),
        "correct_queries": int(details["correct"].sum()),
        "tool_selection_accuracy": float(details["correct"].mean()),
        "per_tool": per_tool,
        "clarification_accuracy": per_tool.get(
            "need_clarification", {}
        ).get("accuracy"),
        "incorrect_cases": mistakes[
            [
                "query",
                "expected_action",
                "actual_action",
                "error",
            ]
        ].to_dict(orient="records"),
        "average_latency_ms": statistics.fmean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "total_evaluation_seconds": time.perf_counter() - total_started,
    }
    return summary, details


def main() -> None:
    """Run the full evaluation and persist measured evidence."""
    summary, details = evaluate_langgraph_router()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    details.to_csv(DETAILS_PATH, index=False)

    print("\nLANGGRAPH ROUTER EVALUATION")
    print(f"Correct: {summary['correct_queries']}/{summary['total_queries']}")
    print(f"Accuracy: {summary['tool_selection_accuracy']:.2%}")
    print(f"Average latency: {summary['average_latency_ms']:.2f}ms")
    print(f"Median latency: {summary['median_latency_ms']:.2f}ms")
    print("\nPER-ACTION BREAKDOWN")
    for action, metrics in summary["per_tool"].items():
        print(
            f"- {action}: {metrics['correct']}/{metrics['total']} "
            f"({metrics['accuracy']:.2%}), "
            f"avg {metrics['average_latency_ms']:.2f}ms"
        )
    print("\nINCORRECT CASES")
    print("None" if not summary["incorrect_cases"] else summary["incorrect_cases"])


if __name__ == "__main__":
    main()
