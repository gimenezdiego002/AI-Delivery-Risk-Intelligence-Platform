# Google Cloud Run deployment

This is the only Phase 10 platform configuration. It contains no credentials.

## 1. Prerequisites requiring the project owner

1. Create or select a Google Cloud project.
2. Enable billing and set a billing budget/alert.
3. Install and authenticate the Google Cloud CLI.
4. Set the project:

```powershell
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

5. Enable required services:

```powershell
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
```

## 2. Create secrets safely

Generate a separate long random backend API key. Do not reuse the OpenAI key.
Create these secrets through the Google Cloud Secret Manager console:

- `delivery-risk-openai-key`
- `delivery-risk-backend-api-key`

Grant the Cloud Run service identity the Secret Manager Secret Accessor role.
Do not place either secret in this repository or the command examples.

## 3. Deploy from the Dockerfile

Run from the repository root:

```powershell
gcloud run deploy delivery-risk-api `
  --source . `
  --region us-east1 `
  --allow-unauthenticated `
  --port 8000 `
  --cpu 1 `
  --memory 1Gi `
  --concurrency 4 `
  --min-instances 0 `
  --max-instances 1 `
  --timeout 120 `
  --set-env-vars "APP_ENV=production,LOG_LEVEL=INFO,LLM_PROVIDER=openai,LLM_MODEL=gpt-4o-mini,LLM_TIMEOUT_SECONDS=30,LLM_MAX_RETRIES=2,LLM_MAX_TOOL_CALLS=3,API_AUTH_ENABLED=true,RATE_LIMIT_ENABLED=true,RATE_LIMIT_REQUESTS=60,RATE_LIMIT_LLM_REQUESTS=10,RATE_LIMIT_WINDOW_SECONDS=60,CORS_ALLOWED_ORIGINS=" `
  --set-secrets "OPENAI_API_KEY=delivery-risk-openai-key:latest,API_KEY=delivery-risk-backend-api-key:latest"
```

`--allow-unauthenticated` makes the HTTPS service reachable, but the
application still requires `X-API-Key` for inference/agent routes. `/health`
and `/ready` remain public.

The selected `max-instances=1` keeps the in-memory limiter within one active
process boundary. It is not a substitute for a distributed limiter.

## 4. Configure probes

In Cloud Run, edit the service revision:

- HTTP startup/readiness probe: `/ready`, port 8000
- HTTP liveness probe: `/health`, port 8000

The startup probe must allow enough time for the Python process to import
pandas/scikit-learn and validate artifacts.

## 5. Required verification

Do not publish the backend API key. Use a local environment variable:

```powershell
$env:DEPLOYED_API_URL = "https://YOUR_SERVICE_URL"
$env:API_KEY = "YOUR_LOCAL_COPY_OF_THE_BACKEND_KEY"
```

Then verify:

```powershell
Invoke-RestMethod "$env:DEPLOYED_API_URL/health"
Invoke-RestMethod "$env:DEPLOYED_API_URL/ready"

Invoke-RestMethod `
  "$env:DEPLOYED_API_URL/orders/be55f985440dddd650b389a55db8e49c/risk" `
  -Headers @{"X-API-Key"=$env:API_KEY}
```

Expected prediction:

```text
late_delivery_probability = 0.8500189058886447
risk_level = high
threshold = 0.5
```

After deployment, run the benchmark with the API key kept only in the local
environment:

```powershell
$env:API_BASE_URL = $env:DEPLOYED_API_URL
$env:BENCHMARK_ENVIRONMENT = "public_warm"
$env:BENCHMARK_REQUEST_COUNT = "10"
python -m src.api.benchmark_endpoints
```

Measure a separate first request after the service has scaled to zero for the
cold-start result.
