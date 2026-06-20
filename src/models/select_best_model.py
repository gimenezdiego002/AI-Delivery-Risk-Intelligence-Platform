"""Business-aware Phase 3 model selection and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def select_best_model(results: dict[str, dict[str, Any]]) -> str:
    """Select highest F1, using recall then precision as tie-breakers.

    F1 is the primary criterion because delivery-risk operations need to catch
    late orders without flooding teams with false alarms. Recall breaks close
    ties in favor of missing fewer genuinely late deliveries.
    """
    if not results:
        raise ValueError("No model results were provided for selection.")
    return max(
        results,
        key=lambda name: (
            results[name]["f1"],
            results[name]["recall"],
            results[name]["precision"],
        ),
    )


def save_best_model(
    *,
    model: Any,
    model_name: str,
    metrics: dict[str, Any],
    features: list[str],
    cutoff_date: str,
    model_path: Path,
    feature_path: Path,
) -> None:
    """Save the fitted pipeline and its versioned feature contract."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_name": model_name,
            "metrics": metrics,
            "model_features": features,
            "cutoff_date": cutoff_date,
            "target": "is_late",
            "selection_rule": "highest F1; recall then precision tie-breakers",
        },
        model_path,
    )
    feature_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "target": "is_late",
                "cutoff_date": cutoff_date,
                "features": features,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
