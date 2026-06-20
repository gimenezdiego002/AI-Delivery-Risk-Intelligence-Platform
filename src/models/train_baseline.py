"""Train and evaluate the Phase 2 Logistic Regression baseline.

The split is chronological because production predictions concern future
orders. A random split would mix the same historical periods across train and
test sets and give an unrealistically easy interpolation task.

Run from the project root with:
    python -m src.models.train_baseline
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.feature_contract import (
    CATEGORICAL_MODEL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_MODEL_FEATURES,
    leakage_audit_table,
    print_leakage_audit,
    validate_model_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "delivery_features.csv"
)
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_baseline.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "phase2_baseline_metrics.json"
AUDIT_PATH = PROJECT_ROOT / "reports" / "phase2_leakage_audit.csv"
DEFAULT_CUTOFF = pd.Timestamp("2018-05-01")


def chronological_split(
    data: pd.DataFrame, cutoff: pd.Timestamp = DEFAULT_CUTOFF
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split earlier orders into train and later orders into test."""
    train = data.loc[data["order_purchase_timestamp"] < cutoff].copy()
    test = data.loc[data["order_purchase_timestamp"] >= cutoff].copy()

    if train.empty or test.empty:
        raise ValueError(f"Cutoff {cutoff.date()} produced an empty split.")
    if train["is_late"].nunique() < 2 or test["is_late"].nunique() < 2:
        raise ValueError("Both chronological splits must contain both classes.")
    return train, test


def create_logistic_pipeline() -> Pipeline:
    """Create preprocessing and class-balanced Logistic Regression steps."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="unknown"),
            ),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_MODEL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_MODEL_FEATURES),
        ]
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        random_state=42,
    )
    return Pipeline(
        steps=[("preprocessing", preprocessing), ("classifier", classifier)]
    )


def train_and_evaluate(
    data: pd.DataFrame, cutoff: pd.Timestamp = DEFAULT_CUTOFF
) -> tuple[Pipeline, dict[str, Any]]:
    """Train the baseline and return the fitted pipeline plus evaluation data."""
    validate_model_features(data)
    train, test = chronological_split(data, cutoff)

    model = create_logistic_pipeline()
    model.fit(train[MODEL_FEATURES], train["is_late"])
    predictions = model.predict(test[MODEL_FEATURES])

    tn, fp, fn, tp = confusion_matrix(
        test["is_late"], predictions, labels=[False, True]
    ).ravel()
    metrics: dict[str, Any] = {
        "cutoff_date": str(cutoff.date()),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_late_rate": float(train["is_late"].mean()),
        "test_late_rate": float(test["is_late"].mean()),
        "accuracy": float(accuracy_score(test["is_late"], predictions)),
        "precision": float(
            precision_score(test["is_late"], predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(test["is_late"], predictions, zero_division=0)
        ),
        "f1": float(f1_score(test["is_late"], predictions, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
    return model, metrics


def _print_metrics(metrics: dict[str, Any]) -> None:
    """Print evaluation metrics in a readable form."""
    print("\nCHRONOLOGICAL SPLIT")
    print(f"Cutoff: {metrics['cutoff_date']}")
    print(
        f"Train: {metrics['train_rows']:,} orders "
        f"({metrics['train_late_rate']:.2%} late)"
    )
    print(
        f"Test:  {metrics['test_rows']:,} orders "
        f"({metrics['test_late_rate']:.2%} late)"
    )
    print("Random splitting was not used; the test set represents future orders.")

    print("\nBASELINE TEST METRICS")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    matrix = metrics["confusion_matrix"]
    print("Confusion matrix:")
    print(f"  True negatives:  {matrix['true_negative']:,}")
    print(f"  False positives: {matrix['false_positive']:,}")
    print(f"  False negatives: {matrix['false_negative']:,}")
    print(f"  True positives:  {matrix['true_positive']:,}")
    print(
        "Accuracy alone is misleading because late orders are rare; a model "
        "can score highly by predicting almost every order as on-time."
    )


def main() -> None:
    """Load features, audit leakage, train, evaluate, and save artifacts."""
    if not FEATURE_DATASET_PATH.exists():
        raise SystemExit(
            "Phase 2 features are missing. Run: "
            ".\\.venv\\Scripts\\python.exe -m src.features.build_features"
        )

    data = pd.read_csv(
        FEATURE_DATASET_PATH,
        parse_dates=["order_purchase_timestamp"],
    )
    print_leakage_audit()

    model, metrics = train_and_evaluate(data)
    _print_metrics(metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_features": MODEL_FEATURES,
            "cutoff_date": metrics["cutoff_date"],
            "target": "is_late",
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    leakage_audit_table().to_csv(AUDIT_PATH, index=False)
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved leakage audit: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
