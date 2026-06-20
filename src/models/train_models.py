"""Train, track, compare, and persist the Phase 3 model candidates.

All candidates use the Phase 2 feature contract and the exact same
chronological cutoff. Keeping the test orders identical makes metric
differences attributable to the models rather than to an easier or harder
sample.

Run from the project root with:
    python -m src.models.train_models
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.features.feature_contract import (
    CATEGORICAL_MODEL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_MODEL_FEATURES,
    validate_model_features,
)
from src.models.evaluate import evaluate_binary_classifier, print_model_metrics
from src.models.mlflow_tracking import configure_mlflow, log_model_experiment
from src.models.select_best_model import select_best_model, save_best_model
from src.models.train_baseline import DEFAULT_CUTOFF, chronological_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
import matplotlib.pyplot as plt  # noqa: E402


FEATURE_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "delivery_features.csv"
)
BASELINE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "logistic_regression_baseline.joblib"
)
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_delivery_risk_model.joblib"
MODEL_FEATURES_PATH = PROJECT_ROOT / "models" / "model_features.json"
METRICS_PATH = PROJECT_ROOT / "reports" / "phase_3_metrics.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "phase_3_model_comparison.md"
IMPORTANCE_CHART_PATH = PROJECT_ROOT / "reports" / "phase_3_feature_importance.png"


def create_tree_preprocessor() -> ColumnTransformer:
    """Create missing-value and categorical handling shared by tree models."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
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
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_MODEL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_MODEL_FEATURES),
        ]
    )


def create_random_forest_pipeline() -> Pipeline:
    """Create a class-balanced, beginner-friendly Random Forest pipeline."""
    return Pipeline(
        steps=[
            ("preprocessing", create_tree_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=18,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def create_xgboost_pipeline(scale_pos_weight: float) -> Pipeline:
    """Create a moderately sized XGBoost pipeline with imbalance weighting."""
    return Pipeline(
        steps=[
            ("preprocessing", create_tree_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=350,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def _original_feature_name(transformed_name: str) -> str:
    """Map one-hot/imputation output names back to a business feature."""
    if transformed_name.startswith("numeric__"):
        name = transformed_name.removeprefix("numeric__")
        return name.removeprefix("missingindicator_")

    name = transformed_name.removeprefix("categorical__")
    for feature in sorted(CATEGORICAL_MODEL_FEATURES, key=len, reverse=True):
        if name == feature or name.startswith(f"{feature}_"):
            return feature
    return name


def extract_feature_importance(model: Pipeline) -> pd.DataFrame:
    """Aggregate transformed tree importance back to original features."""
    preprocessing = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]
    transformed_names = preprocessing.get_feature_names_out()
    importance = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "importance": classifier.feature_importances_,
        }
    )
    importance["feature"] = importance["transformed_feature"].map(
        _original_feature_name
    )
    return (
        importance.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_importance_chart(
    importance: pd.DataFrame, model_name: str, output_path: Path
) -> None:
    """Save a simple chart of the 15 most important business features."""
    top_features = importance.head(15).sort_values("importance")
    fig, axis = plt.subplots(figsize=(10, 7))
    axis.barh(top_features["feature"], top_features["importance"])
    axis.set_title(f"{model_name} — Top Feature Importance")
    axis.set_xlabel("Aggregated model importance")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _model_parameters(model_name: str, model: Pipeline) -> dict[str, Any]:
    """Return the small parameter set that defines each candidate."""
    classifier = model.named_steps["classifier"]
    parameter_names = {
        "logistic_regression": ["class_weight", "max_iter"],
        "random_forest": [
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "class_weight",
        ],
        "xgboost": [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "scale_pos_weight",
        ],
    }[model_name]
    parameters = classifier.get_params()
    return {name: parameters[name] for name in parameter_names}


def _write_report(
    *,
    results: dict[str, dict[str, Any]],
    best_model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_importance: pd.DataFrame,
    importance_model_name: str,
    tracking_uri: str,
) -> None:
    """Write the reproducible Phase 3 Markdown comparison report."""
    metric_rows = []
    for name, metrics in results.items():
        metric_rows.append(
            f"| {name} | {metrics['precision']:.3f} | "
            f"{metrics['recall']:.3f} | {metrics['f1']:.3f} | "
            f"{metrics['roc_auc']:.3f} | {metrics['pr_auc']:.3f} | "
            f"{metrics['false_positive']:,} | {metrics['false_negative']:,} |"
        )

    importance_rows = [
        f"| {row.feature} | {row.importance:.4f} |"
        for row in top_importance.head(10).itertuples()
    ]
    report = f"""# Phase 3 Model Comparison

## Experiment design

- Models: Logistic Regression, Random Forest, and XGBoost.
- Train period: before {DEFAULT_CUTOFF.date()} ({len(train):,} orders).
- Test period: on/after {DEFAULT_CUTOFF.date()} ({len(test):,} orders).
- Train late rate: {train['is_late'].mean():.2%}.
- Test late rate: {test['is_late'].mean():.2%}.
- Decision threshold: 0.50 for every model.
- MLflow tracking store: `{tracking_uri}`.

The same chronological test set was used for every model. A random split was
not used because deployment means predicting genuinely future orders.

## Class imbalance

Late orders are the minority class. Logistic Regression and Random Forest use
balanced class weights; XGBoost uses the training ratio of on-time to late
orders as `scale_pos_weight`. Accuracy is not a selection metric because a
model can appear accurate by predicting almost everything as on-time.

## Test metrics

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | False positives | False negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

Metric meanings:

- Precision: among risk alerts, the fraction that were truly late.
- Recall: among truly late orders, the fraction the model caught.
- F1: a balance between precision and recall.
- ROC-AUC: ranking quality across classification thresholds.
- PR-AUC: minority-class ranking quality, sensitive to false alerts.

## Best model

**{best_model_name}** was selected by highest F1, with recall and then
precision used as tie-breakers. F1 matches the business need to catch delays
without overwhelming operations with false alarms.

## Feature importance

Because the overall winner is not necessarily tree-based, this table reports
importance from the best-performing tree candidate: **{importance_model_name}**.
Its transformed one-hot columns were aggregated back to the original business
features. These importances do not explain the Logistic Regression coefficients.

| Feature | Importance |
|---|---:|
{chr(10).join(importance_rows)}

The chart is saved at `reports/phase_3_feature_importance.png`.

## Known limitations

- Distance is straight-line haversine distance, not carrier-route distance.
- Olist is historical Brazilian marketplace data and may not generalize.
- Class weighting improves recall but can create many false positives.
- Feature importance shows model reliance, not causal impact.
- Aggregated one-hot importance can favor high-cardinality categories.
- No threshold tuning or extensive hyperparameter search was performed.

## Recommended Phase 4

Add SHAP explainability, a reusable prediction explanation function, and the
first scoped tools: `predict_delay_risk(order_id)`, `explain_risk(order_id)`,
and `get_seller_history(seller_id)`.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    """Run the complete Phase 3 comparison workflow."""
    for required_path in (FEATURE_DATASET_PATH, BASELINE_MODEL_PATH):
        if not required_path.exists():
            raise SystemExit(f"Required Phase 2 artifact is missing: {required_path}")

    data = pd.read_csv(
        FEATURE_DATASET_PATH, parse_dates=["order_purchase_timestamp"]
    )
    validate_model_features(data)
    train, test = chronological_split(data, DEFAULT_CUTOFF)
    x_train, y_train = train[MODEL_FEATURES], train["is_late"]
    x_test, y_test = test[MODEL_FEATURES], test["is_late"]

    negative_count = int((~y_train).sum())
    positive_count = int(y_train.sum())
    scale_pos_weight = negative_count / positive_count
    print("CLASS BALANCE")
    print(f"Train late rate: {y_train.mean():.2%}")
    print(f"Test late rate:  {y_test.mean():.2%}")
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.3f}")

    baseline_artifact = joblib.load(BASELINE_MODEL_PATH)
    models: dict[str, Pipeline] = {
        "logistic_regression": baseline_artifact["model"],
        "random_forest": create_random_forest_pipeline(),
        "xgboost": create_xgboost_pipeline(scale_pos_weight),
    }

    results: dict[str, dict[str, Any]] = {}
    importance_tables: dict[str, pd.DataFrame] = {}
    run_ids: dict[str, str] = {}
    tracking_uri = configure_mlflow(PROJECT_ROOT)

    for model_name, model in models.items():
        print(f"\nTraining/evaluating: {model_name}")
        if model_name != "logistic_regression":
            model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics = evaluate_binary_classifier(y_test, predictions, probabilities)
        results[model_name] = metrics
        print_model_metrics(model_name, metrics)

        importance_path = None
        if model_name in {"random_forest", "xgboost"}:
            importance = extract_feature_importance(model)
            importance_tables[model_name] = importance
            importance_path = (
                PROJECT_ROOT
                / "reports"
                / f"{model_name}_feature_importance.csv"
            )
            importance.to_csv(importance_path, index=False)

        run_ids[model_name] = log_model_experiment(
            model_name=model_name,
            model=model,
            parameters={
                **_model_parameters(model_name, model),
                "train_rows": len(train),
                "test_rows": len(test),
                "train_late_rate": round(float(y_train.mean()), 6),
                "test_late_rate": round(float(y_test.mean()), 6),
            },
            metrics=metrics,
            features=MODEL_FEATURES,
            cutoff_date=str(DEFAULT_CUTOFF.date()),
            project_root=PROJECT_ROOT,
            feature_importance_path=importance_path,
        )

    best_model_name = select_best_model(results)
    best_model = models[best_model_name]
    save_best_model(
        model=best_model,
        model_name=best_model_name,
        metrics=results[best_model_name],
        features=MODEL_FEATURES,
        cutoff_date=str(DEFAULT_CUTOFF.date()),
        model_path=BEST_MODEL_PATH,
        feature_path=MODEL_FEATURES_PATH,
    )

    importance_model_name = (
        best_model_name
        if best_model_name in importance_tables
        else max(
            importance_tables,
            key=lambda name: results[name]["f1"],
        )
    )
    selected_importance = importance_tables[importance_model_name]
    save_importance_chart(
        selected_importance, importance_model_name, IMPORTANCE_CHART_PATH
    )

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "cutoff_date": str(DEFAULT_CUTOFF.date()),
                "train_rows": len(train),
                "test_rows": len(test),
                "train_late_rate": float(y_train.mean()),
                "test_late_rate": float(y_test.mean()),
                "best_model": best_model_name,
                "selection_rule": "highest F1; recall then precision tie-breakers",
                "mlflow_tracking_uri": tracking_uri,
                "mlflow_run_ids": run_ids,
                "models": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(
        results=results,
        best_model_name=best_model_name,
        train=train,
        test=test,
        top_importance=selected_importance,
        importance_model_name=importance_model_name,
        tracking_uri=tracking_uri,
    )

    print(f"\nBest model: {best_model_name}")
    print(f"Saved best model: {BEST_MODEL_PATH}")
    print(f"Saved feature contract: {MODEL_FEATURES_PATH}")
    print(f"Saved comparison report: {REPORT_PATH}")
    print(f"Saved feature chart: {IMPORTANCE_CHART_PATH}")
    print(f"MLflow tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
