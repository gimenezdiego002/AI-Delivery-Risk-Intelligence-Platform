"""Local MLflow experiment tracking for Phase 3."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import joblib
import mlflow


EXPERIMENT_NAME = "delivery-risk-phase-3"


def configure_mlflow(project_root: Path) -> str:
    """Configure a local SQLite tracking store and return its URI."""
    database_path = (project_root / "mlflow.db").resolve().as_posix()
    tracking_uri = f"sqlite:///{database_path}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return tracking_uri


def log_model_experiment(
    *,
    model_name: str,
    model: Any,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    features: list[str],
    cutoff_date: str,
    project_root: Path,
    feature_importance_path: Path | None = None,
) -> str:
    """Log model configuration, metrics, feature contract, and fitted artifact."""
    configure_mlflow(project_root)
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_test_cutoff", cutoff_date)
        mlflow.log_param("feature_count", len(features))
        mlflow.log_params(
            {name: str(value) for name, value in parameters.items()}
        )
        mlflow.log_metrics(
            {
                name: float(value)
                for name, value in metrics.items()
                if isinstance(value, (int, float))
            }
        )
        mlflow.log_dict(
            {"model_name": model_name, "features": features},
            "model_features.json",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_path = Path(temporary_directory) / f"{model_name}.joblib"
            joblib.dump(model, artifact_path)
            mlflow.log_artifact(str(artifact_path), artifact_path="model")

        if feature_importance_path and feature_importance_path.exists():
            mlflow.log_artifact(
                str(feature_importance_path), artifact_path="feature_importance"
            )
        return run.info.run_id
