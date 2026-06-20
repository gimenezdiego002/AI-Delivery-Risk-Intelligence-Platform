from pathlib import Path

import kagglehub
import pandas as pd


def main() -> None:
    dataset_path = Path(
        kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    )

    orders = pd.read_csv(
        dataset_path / "olist_orders_dataset.csv",
        parse_dates=[
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )

    delivered_orders = orders.dropna(
        subset=[
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    ).copy()

    delivered_orders["is_late"] = (
        delivered_orders["order_delivered_customer_date"]
        > delivered_orders["order_estimated_delivery_date"]
    )

    print("First five delivered orders:")
    print(delivered_orders.head())
    print(f"\nTotal orders: {len(orders):,}")
    print(f"Delivered orders analyzed: {len(delivered_orders):,}")
    print(f"Late deliveries: {delivered_orders['is_late'].sum():,}")
    print(f"Late-delivery rate: {delivered_orders['is_late'].mean():.2%}")


if __name__ == "__main__":
    main()
