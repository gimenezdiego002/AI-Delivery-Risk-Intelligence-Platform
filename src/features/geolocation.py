"""Geolocation features for delivery-risk modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6_371.0088


def haversine_distance_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    """Return straight-line great-circle distance between coordinate pairs.

    The haversine formula approximates Earth as a sphere. This is useful when
    route data is unavailable, but it is not driving distance: roads, terrain,
    transfer hubs, and carrier routes can make actual travel substantially
    longer. Missing coordinates remain missing rather than being treated as
    zero-distance orders.
    """
    lat1_rad = np.radians(lat1.astype(float))
    lon1_rad = np.radians(lon1.astype(float))
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    haversine_a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    )
    return pd.Series(
        2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(haversine_a)),
        index=lat1.index,
    )


def aggregate_zip_coordinates(geolocation: pd.DataFrame) -> pd.DataFrame:
    """Create one robust median coordinate per Olist zip-code prefix."""
    required_columns = {
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
    }
    missing_columns = required_columns.difference(geolocation.columns)
    if missing_columns:
        raise ValueError(
            f"Geolocation data is missing columns: {sorted(missing_columns)}"
        )

    valid = geolocation.dropna(subset=list(required_columns)).copy()
    valid = valid.loc[
        valid["geolocation_lat"].between(-90, 90)
        & valid["geolocation_lng"].between(-180, 180)
    ]

    return (
        valid.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(
            latitude=("geolocation_lat", "median"),
            longitude=("geolocation_lng", "median"),
        )
    )


def add_distance_feature(
    orders: pd.DataFrame, geolocation: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add seller/customer distance and return explicit match statistics."""
    required_order_columns = {
        "seller_zip_code_prefix",
        "customer_zip_code_prefix",
    }
    missing_columns = required_order_columns.difference(orders.columns)
    if missing_columns:
        raise ValueError(f"Orders are missing columns: {sorted(missing_columns)}")

    zip_coordinates = aggregate_zip_coordinates(geolocation)
    customer_coordinates = zip_coordinates.rename(
        columns={
            "geolocation_zip_code_prefix": "customer_zip_code_prefix",
            "latitude": "customer_latitude",
            "longitude": "customer_longitude",
        }
    )
    seller_coordinates = zip_coordinates.rename(
        columns={
            "geolocation_zip_code_prefix": "seller_zip_code_prefix",
            "latitude": "seller_latitude",
            "longitude": "seller_longitude",
        }
    )

    featured = orders.merge(
        customer_coordinates,
        on="customer_zip_code_prefix",
        how="left",
        validate="many_to_one",
    ).merge(
        seller_coordinates,
        on="seller_zip_code_prefix",
        how="left",
        validate="many_to_one",
    )

    customer_missing = featured["customer_latitude"].isna()
    seller_missing = featured["seller_latitude"].isna()
    featured["seller_customer_distance_km"] = haversine_distance_km(
        featured["seller_latitude"],
        featured["seller_longitude"],
        featured["customer_latitude"],
        featured["customer_longitude"],
    )

    report = {
        "total_orders": len(featured),
        "missing_customer_coordinates": int(customer_missing.sum()),
        "missing_seller_coordinates": int(seller_missing.sum()),
        "missing_either_coordinate": int((customer_missing | seller_missing).sum()),
        "distance_available": int(
            featured["seller_customer_distance_km"].notna().sum()
        ),
    }

    coordinate_columns = [
        "customer_latitude",
        "customer_longitude",
        "seller_latitude",
        "seller_longitude",
    ]
    return featured.drop(columns=coordinate_columns), report
