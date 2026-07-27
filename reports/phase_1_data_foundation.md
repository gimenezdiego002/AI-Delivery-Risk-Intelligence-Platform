# Phase 1 — Data Foundation Report

## Goal

Phase 1 created the clean project foundation for the AI-Powered Delivery Risk
Intelligence Agent.

The main question this phase prepared the project to answer was:

> Which orders are at risk of late delivery, and why?

Phase 1 did not train a model. Its job was to make the raw Olist data loadable,
joinable, and reusable.

## What was created

### Project structure

The repository was organized into a professional data/ML layout:

- `data/raw/` — original Olist CSV files from Kaggle
- `data/processed/` — cleaned and joined datasets
- `notebooks/` — notebooks for exploration
- `src/data/` — reusable data-loading and dataset-building code
- `src/features/` — reusable feature engineering code
- `src/models/` — model training and comparison code
- `src/agent/` — deterministic tools and later agent routing
- `src/api/` — FastAPI service added later
- `src/app/` — Streamlit app added later
- `models/` — saved model artifacts
- `reports/` — metrics, explanations, screenshots, and phase reports

## Main files

### `src/data/load_data.py`

Loads and inspects raw Olist CSV files from `data/raw/`.

It prints:

- table name
- row count
- column count
- column names

This matters because before joining data, you need to confirm the expected CSV
files exist and have the columns your code depends on.

### `src/data/build_dataset.py`

Builds the first joined order-level dataset.

It joins the core Olist tables:

- orders
- order items
- products
- sellers
- customers
- payments, when available
- reviews, when available

It saves:

- `data/processed/delivery_dataset.csv`

## Output artifact

Verified artifact:

- `data/processed/delivery_dataset.csv`

Verified shape:

- `96,476` rows
- `38` columns

Important initial columns include:

- `order_id`
- `seller_id`
- `customer_id`
- `product_category_name`
- `payment_type`
- `delivery_days`
- `estimated_delivery_days`
- `delay_days`
- `is_late`
- `order_month`
- `order_day_of_week`

## Why this phase matters

Machine learning projects fail quickly if the data foundation is messy.

Phase 1 matters because it creates one reliable order-level dataset that later
phases can reuse instead of repeatedly rejoining raw tables in different ways.

This also makes the project easier to explain in interviews:

> I first built a reproducible data foundation that joins the Olist raw tables
> into a clean order-level delivery dataset.

## How to test it

From the project root:

```powershell
.\.venv\Scripts\python.exe -m src.data.load_data
```

Expected result:

- It prints the raw table names, shapes, and columns.
- Missing files should be reported clearly.

Then run:

```powershell
.\.venv\Scripts\python.exe -m src.data.build_dataset
```

Expected result:

- It creates or refreshes `data/processed/delivery_dataset.csv`.

Quick shape check:

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; df=pd.read_csv('data/processed/delivery_dataset.csv'); print(df.shape)"
```

Expected shape:

```text
(96476, 38)
```

## What came next

Phase 2 added geospatial distance and leakage-safe historical seller/category
risk features.
