# AI-Powered Delivery Risk Intelligence Agent

A portfolio-ready data science project that answers:

> Which orders are at risk of late delivery right now, and why?

The project uses the Brazilian E-Commerce (Olist) dataset to build an
order-level late-delivery classifier and, in later phases, an AI agent that can
explain order, seller, and product-category risk.

## Current status: Phase 2 complete

Phase 1 created the reproducible order-level data foundation. Phase 2 adds
geographic distance, leakage-safe seller/category history, an explicit feature
audit, and a chronologically evaluated Logistic Regression baseline.

## Project structure

```text
.
|-- data/
|   |-- raw/                 # Original Olist CSVs (not committed)
|   `-- processed/           # Generated modeling tables (not committed)
|-- notebooks/               # Exploration and experiments
|-- src/
|   |-- data/                # Loading and dataset-building code
|   |-- features/            # Feature engineering and leakage contract
|   |-- models/              # Model training and evaluation
|   |-- agent/               # Future agent tools
|   |-- api/                 # Future FastAPI service
|   `-- app/                 # Future Streamlit application
|-- reports/                 # Figures and written analysis
|-- models/                  # Saved model artifacts (not committed)
|-- Download_Data.py         # Downloads/copies raw data
|-- requirements.txt
`-- README.md
```

## Setup

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Using the virtual environment's Python directly means activation is optional.

## Phase 1 workflow

### 1. Download the raw data

```powershell
.\.venv\Scripts\python.exe .\Download_Data.py
```

This downloads the latest `olistbr/brazilian-ecommerce` dataset through
KaggleHub and copies its CSV files into `data/raw/`.

### 2. Inspect all raw tables

```powershell
.\.venv\Scripts\python.exe -m src.data.load_data
```

The script prints every table's row count, column count, and column names. It
also reports missing or unreadable files clearly instead of failing silently.

### 3. Build the processed dataset

```powershell
.\.venv\Scripts\python.exe -m src.data.build_dataset
```

The output is `data/processed/delivery_dataset.csv`, with one row per delivered
order. Order items, payments, and reviews are aggregated before joining so that
one-to-many relationships do not create duplicate orders.

Initial engineered columns include:

- `delivery_days`: purchase to actual delivery
- `estimated_delivery_days`: purchase to estimated delivery
- `delay_days`: actual delivery minus estimated delivery
- `is_late`: whether `delay_days` is greater than zero
- `order_month` and `order_day_of_week`
- primary product category, product weight, and seller details
- aggregate item, freight, payment, and review information

`delivery_days` and `delay_days` describe outcomes and are not model inputs
because they are unknown when predicting risk.

## Phase 2 workflow

### 1. Build distance and historical features

```powershell
.\.venv\Scripts\python.exe -m src.features.build_features
```

This creates `data/processed/delivery_features.csv`. Seller/customer distance
uses median coordinates for each zip-code prefix and the haversine formula.
It is straight-line distance, not road or carrier-route distance.

Seller and category statistics use only information available before each
order was placed. Delivery outcomes enter history only after delivery, reviews
enter history only after review creation, and the current order never
contributes to its own features. Missing geolocation and cold-start history are
reported and handled as unknown values by the model pipeline.

### 2. Audit features and train the baseline

```powershell
.\.venv\Scripts\python.exe -m src.models.train_baseline
```

The script prints the full leakage audit, uses orders before `2018-05-01` for
training and later orders for testing, trains class-balanced Logistic
Regression, and saves:

- `models/logistic_regression_baseline.joblib`
- `reports/phase2_baseline_metrics.json`
- `reports/phase2_leakage_audit.csv`

Baseline test results:

- Precision: 0.092
- Recall: 0.840
- F1: 0.167
- True positives: 1,323
- False negatives: 252

The high recall means the baseline catches most late deliveries. Its low
precision means it produces many false alarms, which gives Phase 3 a concrete
model-quality problem to improve.

## Dataset source

Brazilian E-Commerce Public Dataset by Olist, distributed through Kaggle. Raw
and processed data are excluded from Git because they are large and reproducible.

## Roadmap

- **Phase 1:** ingestion, validation, joins, and initial features (complete)
- **Phase 2:** leakage-safe features and Logistic Regression baseline (complete)
- **Phase 3:** Random Forest/XGBoost comparison tracked with MLflow
- **Phase 4:** explanations and scoped agent tools
- **Phase 5:** FastAPI, Streamlit, review embeddings, and containerization
