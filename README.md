# AI-Powered Delivery Risk Intelligence Agent

A portfolio-ready data science project that answers:

> Which orders are at risk of late delivery right now, and why?

The project uses the Brazilian E-Commerce (Olist) dataset to build an
order-level late-delivery classifier and, in later phases, an AI agent that can
explain order, seller, and product-category risk.

## Current status: Phase 4 complete

Phase 1 created the reproducible order-level data foundation. Phase 2 added
leakage-safe features and a Logistic Regression baseline. Phase 3 compares that
baseline with Random Forest and XGBoost on the same future test period and
tracks every experiment with MLflow. Phase 4 exposes deterministic prediction,
explanation, seller-history, and similar-order tools without an LLM.

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

## Phase 3 workflow

Install the declared dependencies, then run the comparison:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.models.train_models
```

All models use the Phase 2 cutoff of `2018-05-01`. Logistic Regression and
Random Forest use balanced class weights; XGBoost uses the training-set class
ratio through `scale_pos_weight`.

Phase 3 test results:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.092 | 0.840 | 0.167 | 0.698 | 0.124 |
| Random Forest | 0.112 | 0.081 | 0.094 | 0.607 | 0.088 |
| XGBoost | 0.124 | 0.166 | 0.142 | 0.650 | 0.102 |

Logistic Regression remains the winner by F1. It catches substantially more
late orders, although its many false positives remain a limitation. Results,
run IDs, parameters, and artifacts are tracked in the local `mlflow.db` store.

To inspect experiments in the MLflow UI:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://127.0.0.1:5000` in a browser.

Phase 3 outputs:

- `models/best_delivery_risk_model.joblib`
- `models/model_features.json`
- `reports/phase_3_model_comparison.md`
- `reports/phase_3_metrics.json`
- `reports/phase_3_feature_importance.png`

## Phase 4 tools

The reusable tools live in `src/agent/tools.py`:

- `predict_delay_risk(order_id)` loads the saved winner and returns a risk
  probability at the existing 0.50 threshold.
- `explain_risk(order_id)` ranks that order's Logistic Regression feature
  contributions and builds deterministic, non-causal summary text.
- `get_seller_history(seller_id, as_of_order_id=None)` returns the exact
  leakage-safe seller snapshot created in Phase 2.
- `get_similar_past_orders(order_id, top_n=5)` uses scaled numeric features,
  one-hot categories, and nearest neighbors over orders completed before the
  query was placed.

Example:

```powershell
.\.venv\Scripts\python.exe -c "from src.agent.tools import predict_delay_risk; print(predict_delay_risk('6340164ffcc87a11dd0ad37d2551994c'))"
```

Run the deterministic tests:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agent_tools -v
```

All tools currently operate on order IDs already present in the prepared
historical feature dataset. Predicting a genuinely new live order will require
an online feature-construction path in a later deployment phase.

## Dataset source

Brazilian E-Commerce Public Dataset by Olist, distributed through Kaggle. Raw
and processed data are excluded from Git because they are large and reproducible.

## Roadmap

- **Phase 1:** ingestion, validation, joins, and initial features (complete)
- **Phase 2:** leakage-safe features and Logistic Regression baseline (complete)
- **Phase 3:** Random Forest/XGBoost comparison tracked with MLflow (complete)
- **Phase 4:** deterministic prediction, explanations, history, and similarity tools (complete)
- **Phase 5:** labeled-query tool-routing evaluation without moving business logic into an LLM
- **Phase 6:** FastAPI, Streamlit, performance testing, and containerization
