"""Download the Olist dataset and copy its CSV files into data/raw/."""

from pathlib import Path
from shutil import copy2

import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    """Download the latest dataset version and populate the raw-data folder."""
    downloaded_path = Path(
        kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    )
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = list(downloaded_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {downloaded_path}")

    for source_path in csv_files:
        destination_path = RAW_DATA_DIR / source_path.name
        copy2(source_path, destination_path)
        print(f"Copied: {destination_path.relative_to(PROJECT_ROOT)}")

    print(f"\nRaw data is ready in: {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
