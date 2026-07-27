# Phase 10 Production-Surface Audit

Audit date: 2026-07-27

This is the required read-only audit performed before Phase 10 application
code changes. It describes what the repository actually does at the start of
the phase.

## Executive findings

The existing backend is functionally strong but not ready to expose directly
to the public internet. Its model and deterministic tools are tested, both
agent implementations have measured routing evidence, and Docker runs without
baked secrets. However, the API begins Phase 10 with:

- no API authentication;
- no rate limiting;
- no request or trace IDs;
- no application-owned structured request logging;
- no explicit production error taxonomy;
- no explicit LLM timeout;
- provider retries that are partly implicit and inconsistent;
- no readiness endpoint;
- public OpenAPI documentation in every environment;
- no CI workflow;
- one broad development requirements file used by the API image;
- a measured image size of 872,320,892 bytes;
- no public deployment configuration or monitoring.

The highest-risk behavior is `POST /agent/query`: it catches every
`RouterError` and immediately retries the entire agent up to four times,
including errors that may be permanent. Each internal structured-output step
can also perform its own corrective retry. There is no explicit network
timeout, backoff, retry classification, request identifier, or rate limit.

## 1. Public API endpoints

| Method and route | Deterministic or LLM | Current success contract | Current error behavior |
|---|---|---|---|
| `GET /health` | Deterministic; does not load model/data | `{"status":"ok","model":"logistic_regression","phase":6}` | Framework-level errors only |
| `GET /orders/{order_id}/risk` | Deterministic saved-model inference | `RiskResponse` | Tool `*_not_found` becomes 404; other tool errors become 422; unexpected exceptions can become 500 |
| `GET /orders/{order_id}/explanation` | Deterministic coefficient calculation | `ExplanationResponse` | Same 404/422 mapping; unexpected exceptions can become 500 |
| `GET /sellers/{seller_id}/history` | Deterministic leakage-safe lookup | `SellerHistoryResponse` | Same 404/422 mapping; unexpected exceptions can become 500 |
| `POST /agent/query` | Paid LLM calls plus deterministic tools | `AgentQueryResponse` | Retries any `RouterError` four times, then returns 502 `agent_router_error`; public message contains `str(last_error)` |
| `POST /agent/langgraph/query` | Paid LLM calls plus deterministic tools | `LangGraphAgentQueryResponse` | Catches unexpected exceptions and returns a generic 502 `langgraph_agent_error`; workflow-level safe errors can remain HTTP 200 |

FastAPI also exposes `/docs`, `/redoc`, and `/openapi.json` by default.

There is no HTTP endpoint for `get_similar_past_orders` directly. It is
available through either agent.

## 2. Request and response schemas

The API defines Pydantic success models for health, risk, explanation, seller
history, plain-agent query, and LangGraph-agent query. Successful response
contracts are explicit and should be preserved.

Known tool failures use this general shape:

```json
{
  "ok": false,
  "error": {
    "code": "order_not_found",
    "message": "Safe tool-provided message"
  }
}
```

Plain-agent exhaustion currently uses:

```json
{
  "ok": false,
  "error": {
    "code": "agent_router_error",
    "message": "String form of the final RouterError"
  }
}
```

LangGraph unexpected failure currently uses a generic safe message:

```json
{
  "ok": false,
  "error": {
    "code": "langgraph_agent_error",
    "message": "The LangGraph agent could not process the request."
  }
}
```

FastAPI/Pydantic request validation uses FastAPI's default HTTP 422 `detail`
format. There is no shared error model, request ID, or stable application-wide
error category. Unhandled deterministic exceptions rely on FastAPI's default
500 handling and server logging.

## 3. LLM calls

The LLM is called only by:

- `POST /agent/query`;
- `POST /agent/langgraph/query`;
- standalone live evaluation scripts.

The deterministic prediction, explanation, seller-history, and health
endpoints do not call an LLM.

The plain router and LangGraph share provider access through
`src/agent/router.py`. OpenAI uses `chat.completions.create`; Gemini uses
`models.generate_content`.

## 4. Current timeout behavior

Application code does not set an explicit timeout on OpenAI or Gemini clients
or per-request provider calls. Behavior therefore depends on provider SDK
defaults. That is unsuitable for a public synchronous HTTP service because a
request may occupy a worker much longer than the application intends.

The local performance clients use a 120-second HTTP timeout. This controls the
test client, not the backend provider call.

Streamlit has its own HTTP request timeout, but it does not constrain provider
work after the backend receives a request.

## 5. Current retry behavior

Three different retry concepts are currently mixed:

1. **Structured-output correction:** routing and final-answer functions make
   one corrective LLM call after malformed JSON or schema failure. This is
   intentional application behavior and should remain distinct.
2. **Plain API retry:** `/agent/query` retries the entire `run_agent` call up
   to four times for every `RouterError`, with no classification, delay,
   jitter, or retry count in logs.
3. **Provider SDK retry:** OpenAI/Gemini client defaults may retry internally,
   but the application neither configures nor measures this consistently.

Live evaluation scripts have separate provider retry loops. Those are not the
runtime API policy.

Deterministic tool failures are not explicitly retried, which is correct.

## 6. Authentication and documentation exposure

There is no authentication. Anyone who can reach the service can:

- run model inference;
- retrieve prepared seller/order information;
- trigger paid plain-agent LLM calls;
- trigger paid LangGraph LLM calls.

`/health`, `/docs`, `/redoc`, and `/openapi.json` are public. Production mode
does not currently change this behavior.

## 7. Rate limiting

There is no rate limiter at application or proxy level. A public caller can
generate unbounded paid LLM traffic, subject only to provider quotas and server
capacity. There is no `429` response or `Retry-After` header.

## 8. Logging and observability

The application has no configured logger or request middleware. Uvicorn emits
basic access logs, but the project does not emit structured events with:

- request ID or trace ID;
- application environment;
- endpoint or tool latency;
- selected agent implementation;
- tool names/count;
- model name;
- retry count;
- stable error category.

LangGraph traces include node/action/tool timing and stop reason in the API
response. Plain-agent traces contain decisions and tool results but no timing.
Neither trace is connected to an API request identifier. The deterministic
tools do not emit timing logs.

No current code intentionally logs API keys, authorization headers, full
feature rows, or chain-of-thought. This safe property must be preserved.

## 9. CORS

No CORS middleware is configured.

That is appropriate for the current architecture: Streamlit calls FastAPI
server-side through Python HTTP requests, so a browser does not call FastAPI
directly. Adding permissive browser CORS would increase exposure without
helping the current UI. Phase 10 should keep CORS disabled when the configured
origin list is empty and permit only explicit configured origins if a future
browser client requires it.

## 10. Existing environment variables

Variables currently read by runtime agent code:

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `openai` or `gemini` |
| `OPENAI_API_KEY` | OpenAI secret |
| `OPENAI_BASE_URL` | Optional compatible endpoint |
| `GEMINI_API_KEY` | Gemini secret |
| `LLM_MODEL` | Provider model |
| `LLM_TEMPERATURE` | Generation temperature |
| `LLM_MAX_TOOL_CALLS` | Agent loop cap |

Variables used mainly by evaluation/performance scripts:

- `LLM_EVAL_DELAY_SECONDS`
- `API_BASE_URL`
- `PHASE9_REQUEST_COUNT`

`LLM_JSON_RETRY_LIMIT` appears in `.env.example`, but the router currently
hardcodes one corrective retry and does not read that variable.

Missing Phase 10 settings include application environment, logging level,
network timeout/retry policy, API authentication, rate limiting, and explicit
CORS origins.

`.env` development loading is supported through `python-dotenv`; VS Code is
also configured to inject it into project terminals. `.env` is ignored by Git
and excluded by `.dockerignore`.

## 11. Runtime files

Minimum backend runtime source:

- `src/api/`;
- `src/agent/tools.py`;
- `src/agent/router.py`;
- `src/agent/langgraph_agent.py`;
- package `__init__.py` files;
- `src/features/feature_contract.py`.

Required artifacts:

- `models/best_delivery_risk_model.joblib` — 11,356 bytes;
- `models/model_features.json` — 771 bytes;
- `data/processed/delivery_features.csv` — 54,336,910 bytes.

The API does not require at runtime:

- `delivery_dataset.csv` — 44,902,479 bytes;
- baseline model artifact;
- raw data;
- notebooks;
- training modules;
- MLflow database/runs;
- reports and screenshots;
- Streamlit code;
- evaluation scripts;
- test/performance scripts.

The current Dockerfile copies all of `src/`, `models/`, and `data/processed/`,
so it includes more files than the API needs.

## 12. Dependency audit

The API image currently installs the full `requirements.txt`.

Verified runtime dependency categories from imports:

- FastAPI and Pydantic;
- Uvicorn;
- pandas and NumPy;
- scikit-learn and joblib;
- OpenAI SDK;
- Google GenAI SDK because provider imports are eager;
- python-dotenv;
- LangGraph and its transitive LangChain Core dependency.

Packages present for development/training/UI rather than API runtime:

- `kagglehub`;
- `matplotlib`;
- `seaborn`;
- `jupyter`;
- `mlflow`;
- `xgboost`;
- `pytest`;
- `requests`;
- `streamlit`.

`requests` is used by Streamlit, not FastAPI. Training-only modules inside
`src/models/` import MLflow, XGBoost, and Matplotlib, but the API does not
import those modules.

The saved production artifact is a scikit-learn Logistic Regression pipeline;
XGBoost is not needed to deserialize it.

## 13. Docker baseline and problems

Current verified image size:

```text
872,320,892 bytes
```

Positive baseline properties:

- `python:3.11-slim`;
- dependency layer copied before source for caching;
- `pip --no-cache-dir`;
- unbuffered output and no bytecode;
- binds Uvicorn to `0.0.0.0:8000`;
- `.env`, raw data, MLflow, reports, Git, and `.venv` excluded;
- previous verification found zero baked secret variables.

Problems:

- installs the full development/training/UI dependency set;
- copies both processed datasets and all model artifacts;
- copies training, UI, report-generation, and evaluation source;
- runs as root;
- has no container `HEALTHCHECK`;
- has no readiness check;
- does not declare production settings;
- uses one Uvicorn process with no documented deployment worker policy.

The largest likely package contributors are the notebook stack, MLflow,
Streamlit, XGBoost, Matplotlib, and their transitive dependencies. Exact layer
impact must be measured after building the runtime-only image.

## 14. Existing tests and performance evidence

The pre-Phase-10 repository suite has a verified baseline of:

```text
29 passed
```

Coverage includes deterministic tools, LangGraph orchestration safeguards, the
new LangGraph endpoint, and the existing plain endpoint response contract.
Most LangGraph/API behavior uses mocked LLM responses, which is appropriate
for deterministic CI.

There is no GitHub Actions workflow.

Existing performance tooling:

- `src/api/test_performance.py`: 10 calls per Phase 6 endpoint; average,
  median, p95; no saved file and aborts on first error.
- `src/api/test_phase9_performance.py`: configurable count for prediction,
  plain agent, LangGraph direct, and LangGraph two-tool; saves average, median,
  min/max and client retries, but no p95 or error-rate summary.

The most recent Docker measurement used three requests per case. It is useful
evidence but too small for strong performance conclusions.

## 15. Security risks

Deployment-readiness risks, ordered approximately by impact:

1. Unauthenticated paid LLM endpoints permit cost abuse.
2. No rate limit permits bursts and denial of service.
3. No explicit provider timeout can tie up server capacity.
4. Broad plain-agent retry behavior can multiply cost and latency.
5. Public schema/docs reveal the entire callable API surface.
6. No request IDs make abuse and errors difficult to trace.
7. Plain-agent error messages can include raw provider/router exception text.
8. Running as root increases container impact if the process is compromised.
9. No stable error taxonomy complicates monitoring and client handling.
10. No CI or automated secret scan protects future changes.

This audit found no intentionally hardcoded real key in source. A full source,
image, log, authentication, and rate-limit verification is still required
after implementation. That later check is a deployment-readiness verification,
not a penetration test.

## 16. Deployment blockers

Repository-side blockers:

- centralized validated settings;
- safe logging and correlation IDs;
- authentication and rate limiting;
- explicit timeout/retry policy;
- runtime-only image;
- readiness check;
- deterministic CI;
- production deployment manifest;
- security and performance evidence.

External blockers:

- user-selected host and account;
- billing/free-tier decision;
- platform credentials or connected deployment integration;
- production API key and LLM provider secret;
- desired public domain/URL.

The repository can be prepared without those external values. A public
deployment cannot be honestly claimed until the user authorizes a platform and
the resulting service is exercised over its public URL.

## 17. Contract-preservation plan

Phase 10 can add middleware, dependencies, internal logging, safe exception
mapping, authentication, rate limiting, and `/ready` without changing existing
successful response schemas.

Authentication will intentionally change unauthorized access to protected
routes when enabled. It must default off for local development and on in
validated production configuration. `/health` will remain public.

Detailed timing belongs in logs/internal traces rather than required response
fields. CORS will remain disabled unless explicit origins are configured.

## Audit conclusion

The API is a sound portfolio backend with verified business behavior, but it
is not safe to expose publicly in its current form. Phase 10 should proceed
with additive infrastructure around the existing endpoints and agents. The
model, threshold, feature contract, deterministic calculations, default plain
router, and successful API response contracts do not need to change.
