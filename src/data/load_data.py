"""Inspect the raw Olist CSV files.

Run from the project root with:
    python -m src.data.load_data
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"

EXPECTED_FILES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def load_table(table_name: str, raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load one expected Olist table and raise a clear error if it is missing."""
    if table_name not in EXPECTED_FILES:
        expected = ", ".join(sorted(EXPECTED_FILES))
        raise ValueError(f"Unknown table '{table_name}'. Expected one of: {expected}")

    csv_path = raw_dir / EXPECTED_FILES[table_name]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing '{table_name}' data: {csv_path}\n"
            "Run Download_Data.py or copy the Olist CSV files into data/raw/."
        )

    return pd.read_csv(csv_path)


def inspect_raw_tables(raw_dir: Path = DEFAULT_RAW_DIR) -> bool:
    """Print each expected table's shape and columns; return whether all exist."""
    print(f"Inspecting raw data in: {raw_dir}\n")
    all_files_found = True

    for table_name, filename in EXPECTED_FILES.items():
        csv_path = raw_dir / filename
        if not csv_path.exists():
            all_files_found = False
            print(f"[MISSING] {table_name}: expected {csv_path}")
            continue

        try:
            table = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError) as exc:
            all_files_found = False
            print(f"[ERROR] {table_name}: could not read {csv_path} ({exc})")
            continue

        print(f"[OK] {table_name}: {table.shape[0]:,} rows x {table.shape[1]} columns")
        print(f"     columns: {', '.join(table.columns)}")

    return all_files_found


def main() -> None:
    """Run the raw-data inventory."""
    all_files_found = inspect_raw_tables()
    if not all_files_found:
        print("\nInspection finished with missing or unreadable files.")
    else:
        print("\nAll expected raw tables are present and readable.")


if __name__ == "__main__":
    main()
