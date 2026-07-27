# Phase 6 — FastAPI Service Report

## Goal

Phase 6 added an HTTP API layer on top of the already-built project.

This phase did not retrain the model, rebuild features, or change the Phase 4
tool logic. It simply exposed the existing deterministic tools and agent router
through FastAPI endpoints.

## What was created

### `src/api/main.py`

This file defines the FastAPI app and the production-style HTTP endpoints.

Endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Confirms the API server is running |
| `GET` | `/orders/{order_id}/risk` | Predicts late-delivery risk for one order |
| `GET` | `/orders/{order_id}/explanation` | Explains which model features increased or reduced risk |
| `GET` | `/sellers/{seller_id}/history` | Returns leakage-safe seller history metrics |
| `POST` | `/agent/query` | Runs the Phase 5 natural-language agent loop |

The API uses Pydantic response models so the API docs are structured and easier
to understand.

### `src/api/test_performance.py`

This is a standalone timing script, not a pytest test.

It hits each API endpoint multiple times and prints:

- average latency
- median latency
- p95 latency

This matters because it gives real measured numbers for portfolio/resume claims
instead of guesses.

## Verified non-Docker performance baseline

Measured during Phase 6:

| Endpoint | Average ms | Median ms | P95 ms |
|---|---:|---:|---:|
| `/health` | 16.59 | 9.95 | 62.42 |
| `/orders/{order_id}/risk` | 45.96 | 47.71 | 52.67 |
| `/orders/{order_id}/explanation` | 65.73 | 64.78 | 79.23 |
| `/sellers/{seller_id}/history` | 22.83 | 18.04 | 35.75 |
| `/agent/query` | 3978.96 | 4027.47 | 4762.42 |

The `/agent/query` endpoint is much slower because it calls an external LLM API.
The deterministic endpoints are local and much faster.

## Verified example response

Health check:

```json
{"status":"ok","model":"logistic_regression","phase":6}
```

Risk prediction example:

```json
{
  "ok": true,
  "order_id": "be55f985440dddd650b389a55db8e49c",
  "late_delivery_probability": 0.8500189058886447,
  "risk_level": "high",
  "model_name": "logistic_regression",
  "threshold": 0.5
}
```

## Why this phase matters

Before Phase 6, the model and tools could only be used from Python.

After Phase 6, other software can call the system over HTTP. That is the same
pattern used by real ML products:

```text
frontend or external app → API endpoint → model/tool logic → structured response
```

This made Phase 7 possible, because Streamlit could call the API instead of
importing model code directly.

## How to run it

From the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok","model":"logistic_regression","phase":6}
```

FastAPI docs:

```text
http://127.0.0.1:8000/docs
```

## What came next

Phase 7 added a Streamlit portfolio demo that calls these endpoints over HTTP.
