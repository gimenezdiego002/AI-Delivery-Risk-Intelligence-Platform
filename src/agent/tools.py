"""Deterministic delivery-risk tools built on the saved Phase 3 model.

This module contains no LLM client, prompt, agent router, or external API call.
Each function returns structured Python data so a later interface can consume
it without moving business logic into an LLM.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.feature_contract import CATEGORICAL_MODEL_FEATURES
from src.observability import timed_operation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_delivery_risk_model.joblib"
MODEL_FEATURES_PATH = PROJECT_ROOT / "models" / "model_features.json"
FEATURE_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "delivery_features.csv"
)
RISK_THRESHOLD = 0.50


def _error(code: str, message: str, **context: Any) -> dict[str, Any]:
    """Return a consistent structured error instead of raising to callers."""
    return {
        "ok": False,
        "error": {"code": code, "message": message},
        **context,
    }


@lru_cache(maxsize=1)
def _load_model_bundle() -> dict[str, Any]:
    """Load and validate the existing Phase 3 model artifact once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")
    with timed_operation("model_load", model_name="logistic_regression"):
        bundle = joblib.load(MODEL_PATH)
    required_keys = {"model", "model_name", "model_features"}
    missing_keys = required_keys.difference(bundle)
    if missing_keys:
        raise ValueError(f"Model artifact is missing keys: {sorted(missing_keys)}")
    return bundle


@lru_cache(maxsize=1)
def _load_model_features() -> list[str]:
    """Load the persisted feature contract rather than redefining it here."""
    if not MODEL_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Saved model feature contract not found: {MODEL_FEATURES_PATH}"
        )
    payload = json.loads(MODEL_FEATURES_PATH.read_text(encoding="utf-8"))
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("model_features.json does not contain a feature list.")
    return features


@lru_cache(maxsize=1)
def _load_feature_dataset() -> pd.DataFrame:
    """Load completed order features and timestamps once per process."""
    if not FEATURE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Processed feature dataset not found: {FEATURE_DATASET_PATH}"
        )
    with timed_operation("feature_dataset_load"):
        return pd.read_csv(
            FEATURE_DATASET_PATH,
            parse_dates=[
                "order_purchase_timestamp",
                "order_delivered_customer_date",
                "review_available_at",
            ],
        )


def _validated_order_row(order_id: str) -> tuple[pd.DataFrame | None, dict | None]:
    """Return a one-row frame or a structured validation error."""
    try:
        data = _load_feature_dataset()
        features = _load_model_features()
        model_bundle = _load_model_bundle()
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
        return None, _error("artifact_error", str(exc), order_id=order_id)

    missing_columns = sorted(set(features).difference(data.columns))
    if missing_columns:
        return None, _error(
            "missing_feature_columns",
            "The processed dataset does not satisfy the saved feature contract.",
            order_id=order_id,
            missing_features=missing_columns,
        )
    if model_bundle["model_features"] != features:
        return None, _error(
            "feature_contract_mismatch",
            "The model artifact and model_features.json do not agree.",
            order_id=order_id,
        )

    order = data.loc[data["order_id"] == order_id]
    if order.empty:
        return None, _error(
            "order_not_found",
            f"Order '{order_id}' does not exist in the processed dataset.",
            order_id=order_id,
        )
    if len(order) > 1:
        return None, _error(
            "duplicate_order",
            f"Order '{order_id}' appears more than once in the processed dataset.",
            order_id=order_id,
        )

    missing_values = [
        feature for feature in features if pd.isna(order.iloc[0][feature])
    ]
    if missing_values:
        return None, _error(
            "missing_feature_values",
            "The order has missing required values; prediction was not attempted.",
            order_id=order_id,
            missing_features=missing_values,
        )
    return order, None


def predict_delay_risk(order_id: str) -> dict[str, Any]:
    """Predict late-delivery probability for one known completed order."""
    order, error = _validated_order_row(order_id)
    if error:
        return error

    bundle = _load_model_bundle()
    features = _load_model_features()
    with timed_operation(
        "model_prediction",
        model_name=bundle["model_name"],
    ):
        probability = float(
            bundle["model"].predict_proba(order[features])[:, 1][0]
        )
    return {
        "ok": True,
        "order_id": order_id,
        "late_delivery_probability": probability,
        "risk_level": "high" if probability >= RISK_THRESHOLD else "low",
        "model_name": bundle["model_name"],
        "threshold": RISK_THRESHOLD,
    }


def _map_transformed_feature(
    transformed_name: str, model_features: list[str]
) -> str | None:
    """Map pipeline output names back to one of the approved input features."""
    if transformed_name.startswith("numeric__"):
        name = transformed_name.removeprefix("numeric__")
        name = name.removeprefix("missingindicator_")
        return name if name in model_features else None

    name = transformed_name.removeprefix("categorical__")
    categorical_features = [
        feature for feature in model_features if feature in CATEGORICAL_MODEL_FEATURES
    ]
    for feature in sorted(categorical_features, key=len, reverse=True):
        if name == feature or name.startswith(f"{feature}_"):
            return feature
    return None


def explain_risk(order_id: str, top_n: int = 5) -> dict[str, Any]:
    """Explain model associations, not causes, for one order's prediction."""
    if top_n < 1:
        return _error(
            "invalid_top_n", "top_n must be at least 1.", order_id=order_id
        )

    prediction = predict_delay_risk(order_id)
    if not prediction["ok"]:
        return prediction

    order, error = _validated_order_row(order_id)
    if error:
        return error
    bundle = _load_model_bundle()
    features = _load_model_features()
    model = bundle["model"]
    preprocessing = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]

    transformed = preprocessing.transform(order[features])
    transformed_values = (
        transformed.toarray()[0]
        if hasattr(transformed, "toarray")
        else np.asarray(transformed)[0]
    )
    transformed_names = preprocessing.get_feature_names_out()
    signed_contributions = transformed_values * classifier.coef_[0]

    aggregated = {feature: 0.0 for feature in features}
    for transformed_name, contribution in zip(
        transformed_names, signed_contributions, strict=True
    ):
        original_feature = _map_transformed_feature(transformed_name, features)
        if original_feature is not None:
            aggregated[original_feature] += float(contribution)

    ranked = sorted(
        aggregated.items(), key=lambda item: abs(item[1]), reverse=True
    )[: min(top_n, len(aggregated))]
    explanations = [
        {
            "feature": feature,
            "actual_value": _json_safe_value(order.iloc[0][feature]),
            "direction": (
                "increases_risk" if contribution >= 0 else "decreases_risk"
            ),
            "signed_log_odds_contribution": contribution,
            "approximate_magnitude": abs(contribution),
        }
        for feature, contribution in ranked
    ]

    increases = [
        item["feature"]
        for item in explanations
        if item["direction"] == "increases_risk"
    ]
    decreases = [
        item["feature"]
        for item in explanations
        if item["direction"] == "decreases_risk"
    ]
    summary_parts = [
        f"The model assigned {prediction['risk_level']} late-delivery risk."
    ]
    if increases:
        summary_parts.append(
            "Strongest positive model associations: " + ", ".join(increases) + "."
        )
    if decreases:
        summary_parts.append(
            "Strongest negative model associations: " + ", ".join(decreases) + "."
        )
    summary_parts.append(
        "These are correlational model contributions, not causal effects."
    )

    return {
        "ok": True,
        "order_id": order_id,
        "late_delivery_probability": prediction["late_delivery_probability"],
        "risk_level": prediction["risk_level"],
        "model_name": prediction["model_name"],
        "explanations": explanations,
        "summary": " ".join(summary_parts),
        "caveat": "Model associations are correlational, not causal.",
    }


def _json_safe_value(value: Any) -> Any:
    """Convert pandas/NumPy scalar values into JSON-friendly Python values."""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def get_seller_history(
    seller_id: str, as_of_order_id: str | None = None
) -> dict[str, Any]:
    """Return the exact leakage-safe Phase 2 seller snapshot.

    When ``as_of_order_id`` is supplied, the snapshot is taken at that order's
    purchase time. Otherwise the seller's latest available order-time snapshot
    is returned. The function reuses Phase 2 columns directly, so its history
    cannot drift from the original time-respecting implementation.
    """
    try:
        data = _load_feature_dataset()
    except (FileNotFoundError, ValueError, OSError) as exc:
        return _error("artifact_error", str(exc), seller_id=seller_id)

    seller_orders = data.loc[data["seller_id"] == seller_id].copy()
    if seller_orders.empty:
        return _error(
            "seller_not_found",
            f"Seller '{seller_id}' does not exist in the processed dataset.",
            seller_id=seller_id,
        )

    if as_of_order_id is not None:
        snapshot = seller_orders.loc[seller_orders["order_id"] == as_of_order_id]
        if snapshot.empty:
            return _error(
                "seller_order_not_found",
                f"Order '{as_of_order_id}' is not represented by seller '{seller_id}'.",
                seller_id=seller_id,
                as_of_order_id=as_of_order_id,
            )
    else:
        snapshot = seller_orders.nlargest(1, "order_purchase_timestamp")

    row = snapshot.iloc[0]
    return {
        "ok": True,
        "seller_id": seller_id,
        "as_of_order_id": row["order_id"],
        "history_cutoff": row["order_purchase_timestamp"].isoformat(),
        "historical_order_volume": int(row["seller_historical_order_volume"]),
        "historical_late_rate": _optional_float(
            row["seller_historical_late_rate"]
        ),
        "historical_avg_review_score": _optional_float(
            row["seller_historical_avg_review_score"]
        ),
        "history_source": "precomputed Phase 2 leakage-safe features",
        "leakage_rule": (
            "Only information available strictly before history_cutoff is used; "
            "delivery outcomes require prior delivery and reviews require prior creation."
        ),
    }


def _optional_float(value: Any) -> float | None:
    """Return a normal float or None for an unavailable historical value."""
    return None if pd.isna(value) else float(value)


def get_similar_past_orders(order_id: str, top_n: int = 5) -> dict[str, Any]:
    """Find similar orders completed before the query order was placed."""
    if top_n < 1:
        return _error(
            "invalid_top_n", "top_n must be at least 1.", order_id=order_id
        )

    order, error = _validated_order_row(order_id)
    if error:
        return error
    data = _load_feature_dataset()
    features = _load_model_features()
    query_time = order.iloc[0]["order_purchase_timestamp"]

    candidates = data.loc[
        (data["order_purchase_timestamp"] < query_time)
        & (data["order_delivered_customer_date"] < query_time)
        & (data["order_id"] != order_id)
    ].copy()
    # Explicit safety check: the query must never be its own neighbor.
    candidates = candidates.loc[candidates["order_id"] != order_id]
    if candidates.empty:
        return _error(
            "no_historical_candidates",
            "No orders were completed before this order was placed.",
            order_id=order_id,
        )

    categorical_features = [
        feature for feature in features if feature in CATEGORICAL_MODEL_FEATURES
    ]
    numeric_features = [
        feature for feature in features if feature not in categorical_features
    ]
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median", keep_empty_features=True
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant", fill_value="unknown"
                            ),
                        ),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    candidate_matrix = preprocessing.fit_transform(candidates[features])
    query_matrix = preprocessing.transform(order[features])

    neighbor_count = min(top_n, len(candidates))
    search = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
    search.fit(candidate_matrix)
    distances, positions = search.kneighbors(query_matrix)

    similar_orders = []
    for distance, position in zip(distances[0], positions[0], strict=True):
        candidate = candidates.iloc[int(position)]
        if candidate["order_id"] == order_id:
            continue
        similar_orders.append(
            {
                "order_id": candidate["order_id"],
                "similarity_distance": float(distance),
                "was_late": bool(candidate["is_late"]),
                "order_purchase_timestamp": candidate[
                    "order_purchase_timestamp"
                ].isoformat(),
                "order_delivered_customer_date": candidate[
                    "order_delivered_customer_date"
                ].isoformat(),
            }
        )

    if any(item["order_id"] == order_id for item in similar_orders):
        return _error(
            "self_match_detected",
            "Similarity safety check failed: query order appeared in results.",
            order_id=order_id,
        )
    return {
        "ok": True,
        "order_id": order_id,
        "query_timestamp": query_time.isoformat(),
        "candidate_rule": (
            "Only orders delivered before the query order was placed are eligible."
        ),
        "similar_orders": similar_orders,
    }
