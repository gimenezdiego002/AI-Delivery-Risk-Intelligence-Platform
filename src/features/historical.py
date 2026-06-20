"""Leakage-safe historical seller and category features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _strictly_prior_count(
    data: pd.DataFrame, entity_column: str, time_column: str
) -> pd.Series:
    """Count entity rows whose timestamps are strictly before each query row."""
    result = pd.Series(0, index=data.index, dtype="int64")

    for _, group in data.dropna(subset=[entity_column, time_column]).groupby(
        entity_column, sort=False
    ):
        event_times = np.sort(group[time_column].astype("int64").to_numpy())
        query_times = group[time_column].astype("int64").to_numpy()
        result.loc[group.index] = np.searchsorted(
            event_times, query_times, side="left"
        )

    return result


def _historical_mean(
    data: pd.DataFrame,
    entity_column: str,
    query_time_column: str,
    available_time_column: str,
    value_column: str,
) -> pd.Series:
    """Calculate a mean from values available strictly before each query time."""
    result = pd.Series(np.nan, index=data.index, dtype="float64")
    query_rows = data.dropna(subset=[entity_column, query_time_column])
    history_rows = data.dropna(
        subset=[entity_column, available_time_column, value_column]
    )

    history_groups = {
        entity: group for entity, group in history_rows.groupby(entity_column)
    }
    for entity, queries in query_rows.groupby(entity_column, sort=False):
        history = history_groups.get(entity)
        if history is None:
            continue

        history = history.sort_values(available_time_column)
        available_times = history[available_time_column].astype("int64").to_numpy()
        values = history[value_column].astype(float).to_numpy()
        cumulative_values = np.cumsum(values)

        query_times = queries[query_time_column].astype("int64").to_numpy()
        prior_counts = np.searchsorted(available_times, query_times, side="left")
        has_history = prior_counts > 0
        means = np.full(len(queries), np.nan, dtype=float)
        means[has_history] = (
            cumulative_values[prior_counts[has_history] - 1]
            / prior_counts[has_history]
        )
        result.loc[queries.index] = means

    return result


def add_historical_risk_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add seller/category history that was genuinely available at order time.

    Late-rate and delivery-time history use an order only after its actual
    delivery timestamp, when its outcome is known. Review history uses the
    review creation timestamp. Seller order volume uses only orders placed
    strictly before the current order. Consequently, an order never contributes
    its own outcome, nor any future outcome, to its historical features.
    """
    required_columns = {
        "seller_id",
        "product_category_name",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "is_late",
        "delivery_days",
        "review_score",
        "review_available_at",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Historical feature input is missing: {sorted(missing_columns)}"
        )

    featured = data.copy()
    featured["seller_historical_order_volume"] = _strictly_prior_count(
        featured, "seller_id", "order_purchase_timestamp"
    )
    featured["seller_historical_late_rate"] = _historical_mean(
        featured,
        "seller_id",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "is_late",
    )
    featured["seller_historical_avg_review_score"] = _historical_mean(
        featured,
        "seller_id",
        "order_purchase_timestamp",
        "review_available_at",
        "review_score",
    )
    featured["category_historical_late_rate"] = _historical_mean(
        featured,
        "product_category_name",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "is_late",
    )
    featured["category_historical_avg_delivery_days"] = _historical_mean(
        featured,
        "product_category_name",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "delivery_days",
    )

    return featured
