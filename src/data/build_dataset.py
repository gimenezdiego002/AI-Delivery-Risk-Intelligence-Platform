"""Build the Phase 1 order-level delivery-risk dataset.

Run from the project root with:
    python -m src.data.build_dataset
"""

from pathlib import Path

import pandas as pd

from src.data.load_data import DEFAULT_RAW_DIR, PROJECT_ROOT, load_table


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "delivery_dataset.csv"
DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def _build_item_features(raw_dir: Path) -> pd.DataFrame:
    """Aggregate product, seller, price, and freight data to one row per order."""
    items = load_table("order_items", raw_dir)
    products = load_table("products", raw_dir)
    sellers = load_table("sellers", raw_dir)

    item_details = items.merge(
        products[
            [
                "product_id",
                "product_category_name",
                "product_weight_g",
            ]
        ],
        on="product_id",
        how="left",
        validate="many_to_one",
    ).merge(
        sellers[
            [
                "seller_id",
                "seller_zip_code_prefix",
                "seller_city",
                "seller_state",
            ]
        ],
        on="seller_id",
        how="left",
        validate="many_to_one",
    )

    totals = item_details.groupby("order_id", as_index=False).agg(
        order_item_count=("order_item_id", "count"),
        product_count=("product_id", "nunique"),
        seller_count=("seller_id", "nunique"),
        order_price=("price", "sum"),
        freight_value=("freight_value", "sum"),
        total_product_weight_g=("product_weight_g", "sum"),
    )

    # The highest-priced item represents the order's primary product and seller.
    primary_item = (
        item_details.sort_values(
            ["order_id", "price", "order_item_id"],
            ascending=[True, False, True],
        )
        .drop_duplicates("order_id")
        [
            [
                "order_id",
                "product_id",
                "product_category_name",
                "product_weight_g",
                "seller_id",
                "seller_zip_code_prefix",
                "seller_city",
                "seller_state",
            ]
        ]
    )

    return totals.merge(primary_item, on="order_id", how="left", validate="one_to_one")


def _build_payment_features(raw_dir: Path) -> pd.DataFrame | None:
    """Aggregate optional payment rows to one row per order."""
    try:
        payments = load_table("payments", raw_dir)
    except FileNotFoundError as exc:
        print(f"[OPTIONAL] {exc}")
        return None

    totals = payments.groupby("order_id", as_index=False).agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
        payment_count=("payment_sequential", "count"),
    )
    primary_type = (
        payments.sort_values(
            ["order_id", "payment_value"], ascending=[True, False]
        )
        .drop_duplicates("order_id")[["order_id", "payment_type"]]
    )
    return totals.merge(primary_type, on="order_id", how="left", validate="one_to_one")


def _build_review_features(raw_dir: Path) -> pd.DataFrame | None:
    """Aggregate optional review rows to one row per order."""
    try:
        reviews = load_table("reviews", raw_dir)
    except FileNotFoundError as exc:
        print(f"[OPTIONAL] {exc}")
        return None

    reviews["has_review_comment"] = reviews["review_comment_message"].notna()
    return reviews.groupby("order_id", as_index=False).agg(
        review_score=("review_score", "mean"),
        review_count=("review_id", "count"),
        has_review_comment=("has_review_comment", "max"),
    )


def build_delivery_dataset(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Join Olist tables and create the initial delivery-risk features."""
    orders = load_table("orders", raw_dir)
    customers = load_table("customers", raw_dir)

    for column in DATE_COLUMNS:
        orders[column] = pd.to_datetime(orders[column], errors="coerce")

    dataset = orders.merge(
        customers,
        on="customer_id",
        how="left",
        validate="many_to_one",
    ).merge(
        _build_item_features(raw_dir),
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    payment_features = _build_payment_features(raw_dir)
    if payment_features is not None:
        dataset = dataset.merge(
            payment_features, on="order_id", how="left", validate="one_to_one"
        )

    review_features = _build_review_features(raw_dir)
    if review_features is not None:
        dataset = dataset.merge(
            review_features, on="order_id", how="left", validate="one_to_one"
        )

    # A supervised training label is known only for orders that were delivered.
    required_dates = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    dataset = dataset.dropna(subset=required_dates).copy()

    dataset["delivery_days"] = (
        dataset["order_delivered_customer_date"]
        - dataset["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400
    dataset["estimated_delivery_days"] = (
        dataset["order_estimated_delivery_date"]
        - dataset["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400
    dataset["delay_days"] = (
        dataset["order_delivered_customer_date"]
        - dataset["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86_400
    dataset["is_late"] = dataset["delay_days"] > 0
    dataset["order_month"] = dataset["order_purchase_timestamp"].dt.month
    dataset["order_day_of_week"] = dataset["order_purchase_timestamp"].dt.dayofweek

    dataset = dataset.sort_values("order_purchase_timestamp").reset_index(drop=True)
    if not dataset["order_id"].is_unique:
        raise ValueError("Build failed: processed dataset contains duplicate order IDs.")

    return dataset


def save_delivery_dataset(
    dataset: pd.DataFrame, output_path: Path = DEFAULT_OUTPUT_PATH
) -> None:
    """Save the processed dataset, creating its directory when needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)


def main() -> None:
    """Build, save, and summarize the processed delivery dataset."""
    try:
        dataset = build_delivery_dataset()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Could not build delivery dataset:\n{exc}") from exc

    save_delivery_dataset(dataset)
    print(f"Created: {DEFAULT_OUTPUT_PATH}")
    print(f"Shape: {dataset.shape[0]:,} rows x {dataset.shape[1]} columns")
    print(f"Unique orders: {dataset['order_id'].nunique():,}")
    print(f"Late-delivery rate: {dataset['is_late'].mean():.2%}")


if __name__ == "__main__":
    main()
