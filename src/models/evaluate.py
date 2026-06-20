"""Reusable binary-classification evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true: Any, predictions: Any, probabilities: Any
) -> dict[str, Any]:
    """Calculate minority-class metrics and confusion-matrix values."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 1:
        raise ValueError("Positive-class probabilities must be one-dimensional.")

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[False, True]
    ).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def print_model_metrics(model_name: str, metrics: dict[str, Any]) -> None:
    """Print one model's metrics in a compact, readable block."""
    print(f"\n{model_name}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1:        {metrics['f1']:.3f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.3f}")
    print(
        "  Confusion: "
        f"TN={metrics['true_negative']:,}, "
        f"FP={metrics['false_positive']:,}, "
        f"FN={metrics['false_negative']:,}, "
        f"TP={metrics['true_positive']:,}"
    )
