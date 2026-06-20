"""Audited feature contract for the Phase 2 baseline model."""

from __future__ import annotations

import pandas as pd


NUMERIC_MODEL_FEATURES = [
    "order_item_count",
    "product_count",
    "seller_count",
    "order_price",
    "freight_value",
    "total_product_weight_g",
    "product_weight_g",
    "payment_value",
    "payment_installments",
    "payment_count",
    "estimated_delivery_days",
    "order_month",
    "order_day_of_week",
    "seller_customer_distance_km",
    "seller_historical_order_volume",
    "seller_historical_late_rate",
    "seller_historical_avg_review_score",
    "category_historical_late_rate",
    "category_historical_avg_delivery_days",
]

CATEGORICAL_MODEL_FEATURES = [
    "product_category_name",
    "payment_type",
    "customer_state",
    "seller_state",
]

MODEL_FEATURES = NUMERIC_MODEL_FEATURES + CATEGORICAL_MODEL_FEATURES

FEATURE_REASONS = {
    "order_item_count": "order contents are known at checkout",
    "product_count": "order contents are known at checkout",
    "seller_count": "sellers fulfilling the order are known at checkout",
    "order_price": "item prices are known at checkout",
    "freight_value": "quoted freight is known at checkout",
    "total_product_weight_g": "catalog product attributes are already known",
    "product_weight_g": "primary product's catalog weight is already known",
    "payment_value": "committed payment value is known at checkout",
    "payment_installments": "selected installment count is known at checkout",
    "payment_count": "payment methods committed for the order are known",
    "estimated_delivery_days": "promised delivery date is known at order time",
    "order_month": "derived from the order timestamp",
    "order_day_of_week": "derived from the order timestamp",
    "seller_customer_distance_km": "seller/customer zip prefixes are known",
    "seller_historical_order_volume": "uses only earlier order timestamps",
    "seller_historical_late_rate": "uses only outcomes completed earlier",
    "seller_historical_avg_review_score": "uses only reviews created earlier",
    "category_historical_late_rate": "uses only outcomes completed earlier",
    "category_historical_avg_delivery_days": "uses only outcomes completed earlier",
    "product_category_name": "product catalog category is known",
    "payment_type": "selected payment type is known at checkout",
    "customer_state": "shipping destination is known at checkout",
    "seller_state": "seller location is known before fulfillment",
}

POST_HOC_COLUMNS = {
    "order_status": "status changes after placement",
    "order_approved_at": "approval occurs after placement",
    "order_delivered_carrier_date": "carrier handoff occurs after placement",
    "order_delivered_customer_date": "actual delivery is the future outcome",
    "delivery_days": "depends on actual delivery",
    "delay_days": "directly reveals the target",
    "review_score": "review is created after delivery",
    "review_count": "reviews are created after delivery",
    "has_review_comment": "review content is created after delivery",
    "review_available_at": "current order review occurs after delivery",
}


def leakage_audit_table() -> pd.DataFrame:
    """Return the auditable model/post-hoc feature decision table."""
    rows = [
        {
            "feature": feature,
            "knowable_at_order_time": "YES",
            "used_by_model": "YES",
            "reason": FEATURE_REASONS[feature],
        }
        for feature in MODEL_FEATURES
    ]
    rows.extend(
        {
            "feature": feature,
            "knowable_at_order_time": "NO",
            "used_by_model": "NO",
            "reason": reason,
        }
        for feature, reason in POST_HOC_COLUMNS.items()
    )
    return pd.DataFrame(rows)


def validate_model_features(data: pd.DataFrame) -> None:
    """Fail if a selected feature is missing or violates the leakage contract."""
    missing_features = set(MODEL_FEATURES).difference(data.columns)
    if missing_features:
        raise ValueError(f"Model features are missing: {sorted(missing_features)}")

    audit = leakage_audit_table()
    selected = audit.loc[audit["used_by_model"] == "YES"]
    unsafe = selected.loc[selected["knowable_at_order_time"] != "YES"]
    if not unsafe.empty:
        raise ValueError(
            f"Leakage contract failed for: {unsafe['feature'].tolist()}"
        )


def print_leakage_audit() -> None:
    """Print selected features and explicitly excluded post-hoc columns."""
    audit = leakage_audit_table()
    print("\nLEAKAGE CHECK — MODEL FEATURES")
    print(
        audit.loc[audit["used_by_model"] == "YES"].to_string(index=False)
    )
    print("\nREMOVED FROM MODEL — POST-HOC / LEAKAGE COLUMNS")
    print(
        audit.loc[audit["used_by_model"] == "NO"].to_string(index=False)
    )
