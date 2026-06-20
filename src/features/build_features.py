"""Build the Phase 2 feature dataset without modifying Phase 1 outputs.

Run from the project root with:
    python -m src.features.build_features
"""

from pathlib import Path

import pandas as pd

from src.features.geolocation import add_distance_feature
from src.features.historical import add_historical_risk_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_1_DATASET = PROJECT_ROOT / "data" / "processed" / "delivery_dataset.csv"
GEOLOCATION_PATH = PROJECT_ROOT / "data" / "raw" / "olist_geolocation_dataset.csv"
REVIEWS_PATH = PROJECT_ROOT / "data" / "raw" / "olist_order_reviews_dataset.csv"
PHASE_2_DATASET = PROJECT_ROOT / "data" / "processed" / "delivery_features.csv"


def _load_review_availability() -> pd.DataFrame:
    """Return the time at which each order's complete review data was available."""
    reviews = pd.read_csv(
        REVIEWS_PATH,
        usecols=["order_id", "review_creation_date"],
        parse_dates=["review_creation_date"],
    )
    return (
        reviews.groupby("order_id", as_index=False)
        .agg(review_available_at=("review_creation_date", "max"))
    )


def build_phase_2_features() -> tuple[pd.DataFrame, dict[str, int]]:
    """Add distance and leakage-safe historical features to Phase 1 data."""
    for required_path in (PHASE_1_DATASET, GEOLOCATION_PATH, REVIEWS_PATH):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required Phase 1 input is missing: {required_path}"
            )

    data = pd.read_csv(
        PHASE_1_DATASET,
        parse_dates=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
        ],
    )
    geolocation = pd.read_csv(GEOLOCATION_PATH)
    data, distance_report = add_distance_feature(data, geolocation)
    data = data.merge(
        _load_review_availability(),
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    data = add_historical_risk_features(data)

    if not data["order_id"].is_unique:
        raise ValueError("Phase 2 feature build created duplicate order IDs.")

    return data, distance_report


def _print_missing_feature_report(data: pd.DataFrame) -> None:
    """Report every expected Phase 2 null instead of hiding or zero-filling it."""
    feature_reasons = {
        "seller_customer_distance_km": (
            "seller or customer zip prefix had no valid geolocation match"
        ),
        "seller_historical_late_rate": (
            "no completed order for that seller was available yet"
        ),
        "seller_historical_avg_review_score": (
            "no earlier seller review had been created yet"
        ),
        "category_historical_late_rate": (
            "no completed order for that category was available yet"
        ),
        "category_historical_avg_delivery_days": (
            "no completed order for that category was available yet"
        ),
    }
    print("\nPhase 2 feature missing-value report:")
    for column, reason in feature_reasons.items():
        missing_count = int(data[column].isna().sum())
        print(f"- {column}: {missing_count:,} missing ({reason})")
    print(
        "Missing values are retained as unknown and will be handled by the "
        "model pipeline's median imputer; they are never silently set to zero."
    )


def main() -> None:
    """Build, save, and summarize the Phase 2 feature dataset."""
    try:
        data, distance_report = build_phase_2_features()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Could not build Phase 2 features:\n{exc}") from exc

    PHASE_2_DATASET.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PHASE_2_DATASET, index=False)

    print(f"Created: {PHASE_2_DATASET}")
    print(f"Shape: {data.shape[0]:,} rows x {data.shape[1]} columns")
    print("\nGeolocation match report:")
    for name, count in distance_report.items():
        print(f"- {name}: {count:,}")
    _print_missing_feature_report(data)


if __name__ == "__main__":
    main()
