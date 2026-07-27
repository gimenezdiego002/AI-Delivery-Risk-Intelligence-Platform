"""Evaluate the LLM router's first decision on the labeled Phase 5 set.

This evaluates ``decide_action`` only, not the multi-tool loop. Run from the
project root with:
    python -m src.agent.evaluate_router
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from google.genai import errors as genai_errors
from openai import APIError, RateLimitError

from src.agent.router import RouterError, decide_action


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PATH = PROJECT_ROOT / "tests" / "agent_eval_queries.csv"
RESULTS_PATH = PROJECT_ROOT / "reports" / "phase_5_router_evaluation.json"
DETAILS_PATH = PROJECT_ROOT / "reports" / "phase_5_router_predictions.csv"


def _decide_with_quota_retry(query: str) -> dict[str, Any]:
    """Retry transient provider quota/capacity responses without scoring them."""
    for attempt in range(5):
        try:
            return decide_action(query)
        except (genai_errors.ClientError, genai_errors.ServerError) as exc:
            status_code = getattr(exc, "code", getattr(exc, "status_code", None))
            is_rate_limit = status_code == 429 or "RESOURCE_EXHAUSTED" in str(exc)
            is_temporary_capacity = status_code == 503 or "UNAVAILABLE" in str(exc)
            if not (is_rate_limit or is_temporary_capacity) or attempt == 4:
                raise
            wait_seconds = 15 * (attempt + 1)
            print(f"    API temporarily unavailable; retrying in {wait_seconds}s")
            time.sleep(wait_seconds)
        except (RateLimitError, APIError) as exc:
            status_code = getattr(exc, "status_code", None)
            is_rate_limit = status_code == 429
            is_temporary_capacity = status_code in {500, 502, 503, 504}
            if not (is_rate_limit or is_temporary_capacity) or attempt == 4:
                raise
            wait_seconds = 15 * (attempt + 1)
            print(f"    API temporarily unavailable; retrying in {wait_seconds}s")
            time.sleep(wait_seconds)
    raise RuntimeError("Unreachable quota retry state.")


def evaluate_router(
    evaluation_path: Path = EVALUATION_PATH,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run every query and return summary metrics plus row-level decisions."""
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Evaluation set not found: {evaluation_path}")
    evaluation = pd.read_csv(evaluation_path)
    required_columns = {"query", "expected_tool"}
    missing_columns = required_columns.difference(evaluation.columns)
    if missing_columns:
        raise ValueError(
            f"Evaluation CSV is missing columns: {sorted(missing_columns)}"
        )
    if evaluation.empty:
        raise ValueError("Evaluation set is empty.")

    load_dotenv(override=False)
    evaluation_model = os.getenv("LLM_MODEL", "unknown")
    delay_seconds = float(os.getenv("LLM_EVAL_DELAY_SECONDS", "13.0"))
    rows: list[dict[str, Any]] = []
    completed_queries: set[str] = set()
    if DETAILS_PATH.exists():
        checkpoint = pd.read_csv(DETAILS_PATH)
        if "model" not in checkpoint.columns:
            checkpoint = checkpoint.iloc[0:0]
        else:
            checkpoint = checkpoint.loc[checkpoint["model"] == evaluation_model]
        valid_queries = set(evaluation["query"])
        checkpoint = checkpoint.loc[checkpoint["query"].isin(valid_queries)]
        rows = checkpoint.to_dict(orient="records")
        completed_queries = set(checkpoint["query"])
        if completed_queries:
            print(f"Resuming from checkpoint: {len(completed_queries)} completed")

    for index, item in evaluation.iterrows():
        if item["query"] in completed_queries:
            print(f"[{index + 1:02d}/{len(evaluation)}] checkpointed; skipping")
            continue
        print(f"[{index + 1:02d}/{len(evaluation)}] {item['query']}")
        error_message = None
        try:
            decision = _decide_with_quota_retry(item["query"])
            predicted_tool = (
                decision["tool_name"]
                if decision["status"] == "tool_call"
                else "need_clarification"
            )
        except RouterError as exc:
            predicted_tool = "router_error"
            error_message = str(exc)

        correct = predicted_tool == item["expected_tool"]
        rows.append(
            {
                "query": item["query"],
                "expected_tool": item["expected_tool"],
                "predicted_tool": predicted_tool,
                "correct": correct,
                "error": error_message,
                "model": evaluation_model,
            }
        )
        # Persist each completed row so a long free-tier evaluation leaves a
        # diagnostic checkpoint if the process is interrupted.
        DETAILS_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(DETAILS_PATH, index=False)
        print(
            f"    expected={item['expected_tool']} "
            f"predicted={predicted_tool} correct={correct}"
        )
        if index < len(evaluation) - 1 and delay_seconds > 0:
            time.sleep(delay_seconds)

    details = pd.DataFrame(rows)
    correct_count = int(details["correct"].sum())
    total_count = len(details)
    per_tool = {}
    for expected_tool, group in details.groupby("expected_tool", sort=True):
        tool_correct = int(group["correct"].sum())
        per_tool[expected_tool] = {
            "correct": tool_correct,
            "total": len(group),
            "accuracy": tool_correct / len(group),
        }

    confusion = pd.crosstab(
        details["expected_tool"],
        details["predicted_tool"],
        dropna=False,
    )
    summary = {
        "model": evaluation_model,
        "total_queries": total_count,
        "correct_queries": correct_count,
        "tool_selection_accuracy": correct_count / total_count,
        "per_tool": per_tool,
        "confusion": {
            expected: {
                predicted: int(value)
                for predicted, value in row.items()
                if value
            }
            for expected, row in confusion.to_dict(orient="index").items()
        },
        "router_errors": int((details["predicted_tool"] == "router_error").sum()),
    }
    return summary, details


def print_evaluation(summary: dict[str, Any], details: pd.DataFrame) -> None:
    """Print overall, per-tool, confusion, and mistake diagnostics."""
    print("\nROUTER EVALUATION")
    print(f"Correct: {summary['correct_queries']}/{summary['total_queries']}")
    print(f"Tool-selection accuracy: {summary['tool_selection_accuracy']:.2%}")
    print("\nPER-TOOL BREAKDOWN")
    for tool_name, metrics in summary["per_tool"].items():
        print(
            f"- {tool_name}: {metrics['correct']}/{metrics['total']} "
            f"({metrics['accuracy']:.2%})"
        )
    print("\nCONFUSION MATRIX (rows=expected, columns=predicted)")
    print(
        pd.crosstab(
            details["expected_tool"], details["predicted_tool"], dropna=False
        ).to_string()
    )
    mistakes = details.loc[~details["correct"]]
    print("\nMISTAKES")
    if mistakes.empty:
        print("None")
    else:
        print(
            mistakes[["query", "expected_tool", "predicted_tool", "error"]]
            .to_string(index=False)
        )


def main() -> None:
    """Run, persist, and print the measured routing evaluation."""
    summary, details = evaluate_router()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    details.to_csv(DETAILS_PATH, index=False)
    print_evaluation(summary, details)
    print(f"\nSaved summary: {RESULTS_PATH}")
    print(f"Saved row-level predictions: {DETAILS_PATH}")


if __name__ == "__main__":
    main()
