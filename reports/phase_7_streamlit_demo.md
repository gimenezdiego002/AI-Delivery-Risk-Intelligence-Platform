# Phase 7 — Streamlit Demo Report

## Goal

Phase 7 added a visual portfolio demo for the project.

The Streamlit app is intentionally a frontend only. It does not import the model,
agent tools, or router directly. Instead, it calls the FastAPI service over HTTP.

This keeps the project architecture clean:

```text
Streamlit UI → FastAPI API → model/tools/router → dataset/model artifacts
```

## What was created

### `src/app/streamlit_app.py`

This file defines the Streamlit dashboard.

It includes:

- API health indicator in the sidebar
- Order Risk tab
- Risk Explanation tab
- Seller History tab
- Ask the Agent tab
- dark visual styling
- plain-English error messages
- footer with real project metrics

The app uses `requests` to call FastAPI endpoints.

Confirmed separation:

- The Streamlit file does not import `src.agent.tools`
- The Streamlit file does not import `src.agent.router`
- The Streamlit file does not load the model directly

## UI sections

### Sidebar

Shows:

- project name
- one-line description
- API status

If FastAPI is running, it shows:

```text
API online
```

If FastAPI is not running, it shows:

```text
API offline
```

### Order Risk tab

Input:

- `order_id`

Calls:

- `GET /orders/{order_id}/risk`

Displays:

- HIGH or LOW risk
- probability as a percentage
- model name

### Risk Explanation tab

Input:

- `order_id`

Calls:

- `GET /orders/{order_id}/explanation`

Displays:

- feature name
- whether the feature increased or reduced risk
- approximate magnitude
- deterministic plain-language summary

Visible caveat:

```text
These are correlational signals, not causal explanations
```

### Seller History tab

Input:

- `seller_id`

Calls:

- `GET /sellers/{seller_id}/history`

Displays:

- historical order volume
- historical late rate
- historical average review score

The late-rate display uses risk colors:

- red above 15%
- yellow from 8% to 15%
- green below 8%

### Ask the Agent tab

Input:

- natural-language query

Calls:

- `POST /agent/query`

Displays:

- final answer
- collapsible agent trace showing which tools were called

Warning shown in the UI:

```text
Agent queries call a paid LLM API. Each query costs a small amount.
```

## Demo screenshots

Saved screenshots:

- `reports/demo_screenshots/order_risk_high.png`
- `reports/demo_screenshots/risk_explanation.png`
- `reports/demo_screenshots/seller_history.png`
- `reports/demo_screenshots/agent_loopback_trace.png`

Demo instructions and useful IDs:

- `reports/demo_notes.md`

## Footer metrics

The footer displays:

```text
Built on Olist Brazilian E-Commerce dataset · Logistic Regression · 96,476 orders · 40/40 agent routing accuracy
```

These metrics come from verified project artifacts:

- `data/processed/delivery_features.csv`
- `models/best_delivery_risk_model.joblib`
- `reports/phase_5_router_evaluation.json`

## How to run it

Terminal 1 — start FastAPI:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

Terminal 2 — start Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## Why this phase matters

Phase 7 turns the backend ML/agent work into something visible and demoable.

For interviews, this matters because you can show:

- a real model prediction
- a deterministic explanation
- seller-level operational context
- an LLM agent calling tools
- the trace of which tools were used

That is much stronger than only showing notebooks or code.

## What came next

Phase 8 containerized the FastAPI backend with Docker.
