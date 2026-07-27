# Phase 8 — Docker Containerization Report

## Goal

Phase 8 containerized the FastAPI backend so the API can run in a reproducible
environment.

This phase did not retrain the model, rebuild data, or change source logic in
`src/agent/`, `src/api/`, or `src/app/`.

## What was created

### `Dockerfile`

The Dockerfile builds a FastAPI-only image.

Key decisions:

- Uses `python:3.11-slim`
  - Smaller than the full Python image but still stable for pandas and
    scikit-learn.
- Sets `/app` as the working directory.
- Copies `requirements.txt` before source files.
  - This helps Docker cache the dependency installation layer.
- Copies only runtime files:
  - `src/`
  - `models/`
  - `data/processed/`
- Exposes port `8000`.
- Starts FastAPI with:

```text
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

`0.0.0.0` is required inside Docker because binding to `127.0.0.1` would only
listen inside the container. Binding to `0.0.0.0` allows the host machine to
reach the container through `-p 8000:8000`.

### `.dockerignore`

The `.dockerignore` prevents unnecessary or sensitive files from entering the
Docker build context.

Excluded:

- `.env` — prevents secrets/API keys from being baked into the image
- `.venv/` — local virtual environment is large and machine-specific
- `data/raw/` — large raw Kaggle files are not needed for inference
- `mlflow.db` — experiment history is not needed at runtime
- `reports/` — documentation/screenshots are not runtime dependencies
- `**/__pycache__/` — generated Python caches
- `.git/` — Git history is not needed in the image
- `*.pyc` — compiled Python artifacts

## Environment variables

The image does not contain `.env`.

The router already reads LLM settings from environment variables:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `LLM_PROVIDER`
- `LLM_MODEL`
- `LLM_TEMPERATURE`
- `LLM_MAX_TOOL_CALLS`

At runtime, pass `.env` like this:

```powershell
docker run --env-file .env -p 8000:8000 delivery-risk-api
```

This keeps secrets outside the image.

## Build command

```powershell
docker build -t delivery-risk-api .
```

Verified result:

```text
Build completed successfully.
```

## Run command

For normal use:

```powershell
docker run --env-file .env -p 8000:8000 delivery-risk-api
```

For testing with a named container:

```powershell
docker run -d --name delivery-risk-api-test --env-file .env -p 8000:8000 delivery-risk-api
```

Verified container status:

```text
delivery-risk-api-test Up ... 0.0.0.0:8000->8000/tcp
```

## Verified health check

URL:

```text
http://localhost:8000/health
```

Response:

```json
{"status":"ok","model":"logistic_regression","phase":6}
```

## Verified prediction match

Test order:

```text
be55f985440dddd650b389a55db8e49c
```

Non-Docker prediction:

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

Docker prediction:

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

The outputs match exactly.

## Security verification

Command used:

```powershell
docker run --rm delivery-risk-api env
```

Result:

```text
No OPENAI/GEMINI/API_KEY/LLM variables appeared in the image environment.
```

Additional check:

```powershell
docker run --rm delivery-risk-api sh -c "if [ -f /app/.env ]; then echo ENV_FILE_FOUND; else echo ENV_FILE_NOT_FOUND; fi"
```

Result:

```text
ENV_FILE_NOT_FOUND
```

This confirms `.env` is not inside the built image.

## Dockerized performance results

Measured with:

```powershell
.\.venv\Scripts\python.exe -m src.api.test_performance
```

Against:

```text
http://127.0.0.1:8000
```

| Endpoint | Average ms | Median ms | P95 ms |
|---|---:|---:|---:|
| `/health` | 35.99 | 19.72 | 186.04 |
| `/orders/{order_id}/risk` | 86.79 | 52.84 | 411.48 |
| `/orders/{order_id}/explanation` | 81.68 | 72.92 | 179.52 |
| `/sellers/{seller_id}/history` | 61.54 | 19.33 | 422.89 |
| `/agent/query` | 5336.65 | 5098.78 | 7562.97 |

## Performance comparison versus local Phase 6

| Endpoint | Local avg ms | Docker avg ms | Interpretation |
|---|---:|---:|---|
| `/health` | 16.59 | 35.99 | Slightly slower, still fast |
| `/orders/{order_id}/risk` | 45.96 | 86.79 | Slower, still acceptable |
| `/orders/{order_id}/explanation` | 65.73 | 81.68 | Slightly slower |
| `/sellers/{seller_id}/history` | 22.83 | 61.54 | Slower, still acceptable |
| `/agent/query` | 3978.96 | 5336.65 | Slower; dominated by LLM API latency |

The Dockerized API is slower, but not meaningfully broken. Some overhead is
expected from Docker networking and container runtime. The largest delay is
still from the external LLM call used by `/agent/query`.

## Important observation

The Docker image is larger and slower to build than necessary because the current
`requirements.txt` includes development and UI dependencies such as:

- Jupyter
- MLflow
- XGBoost
- Streamlit

For a future cleanup phase, create a smaller `requirements-api.txt` containing
only the dependencies needed by the FastAPI runtime.

## Streamlit container decision

Recommendation:

Keep Streamlit local for now and point it at the Dockerized FastAPI API.

Why:

- The backend is the production-like service.
- The UI is a portfolio/demo layer.
- It is easier to explain clean architecture:

```text
local Streamlit demo → Dockerized FastAPI API → model/tools/router
```

Containerizing Streamlit too is possible, but not necessary before an evaluation
deadline.

## How to stop the test container

```powershell
docker stop delivery-risk-api-test
```

Optional cleanup:

```powershell
docker rm delivery-risk-api-test
```

## What should come next

Recommended next step:

Update the README with Phase 6–8 status, Docker commands, and demo instructions.

After that, the most useful portfolio improvement would be deployment or a
smaller API-specific Docker image.
