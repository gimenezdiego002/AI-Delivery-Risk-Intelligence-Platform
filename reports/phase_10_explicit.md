# Phase 10 — Production Hardening and Deployment Readiness

## Final status

**Implementation-complete, deployment-pending.**

All repository-side work and local verification applicable without cloud
credentials is complete. Phase 10 is not marked fully complete because Cloud
Run has not been deployed, GitHub-hosted CI has not executed, and public
monitoring/latency/security checks therefore cannot exist yet.

## 1. Production audit

The pre-change audit is in `phase_10_production_audit.md`. It mapped seven
public routes, separated deterministic and paid LLM paths, and identified the
main risks: no centralized settings, no authentication/rate limiting,
unbounded SDK/application retry overlap, inconsistent public errors, limited
request correlation, a development-heavy image, gitignored runtime artifacts
missing from clean builds, and no CI/deployment configuration.

## 2. Runtime configuration

`src/api/config.py` loads `.env` for local development and process environment
variables in deployment, validates them once, and stores secrets as Pydantic
`SecretStr`.

| Setting | Type / constraint | Safe default |
|---|---|---|
| `APP_ENV` | development/test/production | development |
| `LOG_LEVEL` | standard Python levels | INFO |
| `LLM_PROVIDER` | openai/gemini | openai |
| `OPENAI_API_KEY`, `GEMINI_API_KEY`, `API_KEY` | secret, no real default | none |
| `LLM_MODEL` | non-empty string | gpt-4o-mini |
| `LLM_TEMPERATURE` | 0–2 | 0 |
| `LLM_TIMEOUT_SECONDS` | >0, <=300 | 30 |
| `LLM_MAX_RETRIES` | 0–5 | 2 |
| `LLM_JSON_RETRY_LIMIT` | 0–2 | 1 |
| `LLM_MAX_TOOL_CALLS` | 1–10 | 3 |
| `API_AUTH_ENABLED` | boolean | false |
| `RATE_LIMIT_ENABLED` | boolean | false |
| `RATE_LIMIT_REQUESTS` | positive integer | 60 |
| `RATE_LIMIT_LLM_REQUESTS` | positive integer | 10 |
| `RATE_LIMIT_WINDOW_SECONDS` | positive integer | 60 |
| `CORS_ALLOWED_ORIGINS` | explicit CSV origins | empty |

Production refuses to start unless authentication, the backend API key, and
the selected provider key exist. Production docs are disabled.

## 3. Logging, IDs, traces, and latency

Production logs are one JSON object per event; development logs remain readable.
An actual shape is:

```json
{
  "timestamp": "2026-07-27T06:54:27.051762+00:00",
  "level": "INFO",
  "event": "request_completed",
  "request_id": "generated-uuid",
  "trace_id": null,
  "app_env": "production",
  "http_method": "GET",
  "route": "/orders/{order_id}/risk",
  "status_code": 200,
  "latency_ms": 9.839
}
```

Each request gets a UUID unless a safe 1–128-character `X-Request-ID` is
supplied. The ID is returned in the response header. Agent routes also receive
a trace ID propagated into router/graph/tool/provider logs. This exposes
actions, tools, status, stop reason, retries, and elapsed time—not hidden
reasoning.

Monotonic timing covers the HTTP request, model/data load, prediction,
deterministic tools, individual provider calls, and total plain/LangGraph
execution.

## 4. Error taxonomy

The internal categories are:

`validation_error`, `authentication_error`, `rate_limit_error`,
`order_not_found`, `seller_not_found`, `tool_execution_error`,
`model_loading_error`, `llm_timeout`, `llm_rate_limit`,
`llm_invalid_response`, `llm_provider_error`, `agent_max_steps_reached`, and
`internal_error`.

Public messages are stable and safe. Known tool “not found” contracts remain
compatible. Unknown exceptions become a generic 500 without stack traces or
exception strings.

## 5. Timeout and retry policy

- Each provider request has the configured explicit timeout.
- Provider SDK automatic retries are disabled.
- The application makes at most `1 + LLM_MAX_RETRIES` network attempts.
- Only timeout, rate-limit, connection, and 5xx failures retry.
- Backoff is bounded exponential delay with jitter.
- 4xx client errors and deterministic-tool failures do not retry.
- The one structured-output correction retry is counted separately from
  network attempts.
- Exhaustion returns a categorized safe error and logs retry count.

Mocked tests cover timeouts, 429s, transient server failures, malformed JSON,
and permanent client failures without paid calls.

## 6. Authentication, rate limiting, and CORS

Inference and both agent routes require `X-API-Key` when enabled. `/health` and
`/ready` are public. Key comparison is constant-time, keys are never logged,
and production docs are disabled.

The controlled rate test returned `200, 200, 429` and a `Retry-After` header.
The LLM bucket can be stricter than the general bucket. The in-memory design is
honestly limited to one process/container; Cloud Run is capped at one instance.

CORS middleware is absent when no origins are configured. The current
Streamlit app makes server-side Python HTTP requests, so browser CORS is
unnecessary. Wildcard origins are rejected.

## 7. Dependencies and Docker

`requirements-api.txt` contains only FastAPI/Uvicorn/Pydantic/dotenv,
pandas/NumPy/scikit-learn/joblib, OpenAI/Google provider SDKs, and LangGraph.
A brand-new temporary environment installed it and imported the API
successfully.

The Docker image:

- uses Python 3.11 slim;
- reuses dependency layers;
- copies only runtime source;
- restores checksum-documented model/data artifacts;
- runs as UID 10001;
- includes a `/ready` health check;
- supports Cloud Run's injected `PORT`;
- contains no `.env`, raw data, reports, tests, notebooks, Streamlit, MLflow,
  or training packages.

Measured results:

| Measurement | Result |
|---|---:|
| Old image | 872,320,892 bytes |
| Final image | 205,317,857 bytes |
| Reduction | 76.46% |
| Initial optimized cold build | 82.259 s |
| Final cached build | 6.048 s |
| Final startup to readiness | 5.273 s |

The final container reproduced the verified prediction exactly:
`0.8500189058886447`, `high`, threshold `0.5`.

## 8. Health and readiness

`/health` remains the Phase 6-compatible response:

```json
{"status":"ok","model":"logistic_regression","phase":6}
```

`/ready` loads/checks the model artifact, 23-feature contract, and inference
dataset without contacting an LLM:

```json
{"status":"ready","model":"logistic_regression","feature_count":23,"llm_checked":false}
```

## 9. Tests, CI, and live evaluations

Local deterministic equivalent:

- `pytest`: **63 passed**, 2 dependency deprecation warnings;
- compileall: passed;
- credential scan: passed across 118 files;
- runtime-only clean environment import: passed;
- final Docker build: passed;
- final container health: healthy.

CI files:

- `.github/workflows/ci.yml`: normal deterministic tests, credential check,
  artifact checksum/restore, Docker build, and non-root check—no paid calls.
- `.github/workflows/live-agent-evaluation.yml`: manual, secret-dependent,
  provider-backed evaluations and report upload. It deletes the Phase 5
  checkpoint in the ephemeral runner so the run is genuinely live.

**GitHub-hosted CI has not run.** Only its local equivalent passed.

Fresh OpenAI `gpt-4o-mini` evaluation:

- plain first-tool routing: **40/40**;
- LangGraph first-tool routing: **40/40**;
- LangGraph clarification: **8/8**;
- LangGraph full adversarial workflows: **15/15**.

A first fresh plain run inside the network-restricted sandbox failed 0/40 at
the provider boundary. It is preserved separately and was not mislabeled as a
routing result. The approved-network live run passed 40/40.

One additional plain conditional agent request returned
`llm_invalid_response`; the next request succeeded with
`predict_delay_risk -> explain_risk`. This is disclosed provider-format
variability. It did not affect the fresh 40/40 first-tool evaluation.

## 10. Security verification

The deployment-readiness checks found:

- `.env` ignored and untracked;
- no `.env` in the image;
- zero secret variable names in image config/history;
- no actual OpenAI key or auth headers in container logs;
- request/trace IDs and latency present;
- non-root UID 10001;
- unauthenticated inference 401;
- authenticated inference 200;
- production docs 404;
- rate limiting 429 with `Retry-After`;
- no observed production traceback.

See `phase_10_security_verification.md`. This is not a penetration test.

## 11. Deployment and monitoring

Cloud Run was selected over Render, Railway, and Fly.io because the measured
runtime memory needs more margin than common 512 MB starter tiers, while Cloud
Run supports a 1 GiB container, scale-to-zero, Secret Manager, managed HTTPS,
health probes, structured logs, and a one-instance cap.

Repository preparation:

- `.gcloudignore`
- `deploy/cloud-run/README.md`
- runtime `PORT` support
- Secret Manager mapping instructions
- 1 CPU / 1 GiB / concurrency 4 / min 0 / max 1 / timeout 120 seconds

The public blocker is exact and user-owned: a billed Google Cloud project,
authenticated `gcloud`, enabled APIs, and Secret Manager values. No public URL,
public health result, public authorized prediction, public agent trace, public
429, or public cold/warm latency is claimed.

After deployment, Cloud Monitoring should check public `/health` every five
minutes and alert by email after two consecutive failures. It is prepared but
not activated because no public URL exists.

The Streamlit recommendation is to keep it local for interviews and point its
server-side requests at the deployed API. Its API URL and key are configurable
without changing the UX.

## 12. Files created or modified

Production code/configuration:

- `.env.example`, `.gitignore`, `.dockerignore`, `.gcloudignore`
- `Dockerfile`, `requirements-api.txt`
- `src/api/config.py`, `logging_config.py`, `request_context.py`,
  `errors.py`, `security.py`, `rate_limit.py`, `cors.py`, `readiness.py`,
  `benchmark_endpoints.py`, and `main.py`
- `src/observability.py`
- `src/agent/router.py`, `src/agent/langgraph_agent.py`,
  `src/agent/tools.py` (instrumentation/configuration only; model/tool math
  unchanged)
- `src/app/streamlit_app.py` (deployment configuration only)
- `scripts/check_secrets.py`
- `.github/workflows/ci.yml`
- `.github/workflows/live-agent-evaluation.yml`
- `artifacts/runtime_artifacts.tar.gz`, checksum, and documentation
- `deploy/cloud-run/README.md`

Tests:

- `tests/test_api_config.py`
- `tests/test_logging_config.py`
- `tests/test_request_context.py`
- `tests/test_api_errors.py`
- `tests/test_llm_resilience.py`
- `tests/test_api_auth.py`
- `tests/test_rate_limit.py`
- `tests/test_cors_config.py`
- `tests/test_readiness.py`

Documentation/evidence:

- `README.md`, `reports/project_phase_status.md`
- all `reports/phase_10_*.md`
- Phase 10 JSON/CSV evaluation and performance artifacts
- `reports/phase_10_api_workflow_examples.json`

## 13. Completion and Phase 11

Phase 10 is approximately **90% complete**: all repository and local-container
work is verified, while the cloud-owned deployment/hosted-CI/monitoring/public
measurement slice remains. The overall planned portfolio project through this
production milestone is approximately **96% complete**. These percentages are
scope estimates, not measured model metrics.

Phase 11 should be narrowly scoped to:

1. user completes Google Cloud account/project/billing/secret setup;
2. deploy the prepared image to Cloud Run;
3. execute the full public verification checklist;
4. measure public cold/warm performance;
5. activate and test uptime alerts;
6. push/open the repository so GitHub-hosted CI actually runs;
7. optionally deploy Streamlit only after deciding that a public UI is worth
   the extra secret/platform maintenance.

Do not add RAG, memory, embeddings, or another model before deployment evidence
is complete.
