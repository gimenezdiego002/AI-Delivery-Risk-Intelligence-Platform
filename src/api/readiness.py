"""Readiness checks for local, container, and hosted deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.agent.tools import (
    FEATURE_DATASET_PATH,
    MODEL_FEATURES_PATH,
    MODEL_PATH,
    _load_model_bundle,
    _load_model_features,
)
from src.api.errors import AppError, ErrorCategory


def verify_runtime_readiness(
    *,
    model_path: Path = MODEL_PATH,
    feature_contract_path: Path = MODEL_FEATURES_PATH,
    feature_dataset_path: Path = FEATURE_DATASET_PATH,
    model_loader: Callable[[], dict[str, Any]] = _load_model_bundle,
    feature_loader: Callable[[], list[str]] = _load_model_features,
) -> dict[str, Any]:
    """Verify required local artifacts without calling an external provider."""
    missing = [
        label
        for label, path in (
            ("model", model_path),
            ("feature_contract", feature_contract_path),
            ("feature_dataset", feature_dataset_path),
        )
        if not path.is_file()
    ]
    if missing:
        raise AppError(
            ErrorCategory.MODEL_LOADING,
            503,
            internal_context={"missing_artifacts": missing},
        )

    try:
        bundle = model_loader()
        features = feature_loader()
    except Exception as exc:
        raise AppError(
            ErrorCategory.MODEL_LOADING,
            503,
            internal_context={"exception_type": type(exc).__name__},
        ) from exc

    if bundle.get("model_name") != "logistic_regression" or not features:
        raise AppError(
            ErrorCategory.MODEL_LOADING,
            503,
            internal_context={"reason": "invalid_artifact_contract"},
        )
    return {
        "status": "ready",
        "model": "logistic_regression",
        "feature_count": len(features),
        "llm_checked": False,
    }
