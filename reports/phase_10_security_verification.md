# Phase 10 Deployment-Readiness Security Verification

This is a focused deployment-readiness check, **not a penetration test or a
formal security audit**.

## Result

The final local Docker image passed the repository-side security checks. Public
Cloud Run checks remain pending because no service has been deployed.

| Check | Measured result |
|---|---|
| `.env` ignored and untracked | Pass |
| Credential scan | Pass across 118 tracked/unignored files |
| `/app/.env` in image | Absent |
| Secret variable names in image config/history | 0 |
| Secret values in container logs | Not found |
| Authorization or `X-API-Key` headers in logs | Not found |
| Container identity | `uid=10001(appuser)`, non-root |
| Public `/health` | 200 |
| Public `/ready` | 200 |
| Inference without key | 401 |
| Inference with valid key | 200 |
| Production `/docs` | 404 |
| Rate-limit sequence | 200, 200, 429 |
| `Retry-After` on 429 | Present (`58` seconds in the controlled run) |
| Request ID in logs | Present |
| Trace ID in agent logs | Present |
| Latency in logs | Present |
| Python traceback in production logs/responses | Not observed; tests cover safe 500 mapping |

## Secret handling

- Real credentials have no hardcoded default.
- `.env` is used for local development only and is excluded by Git and Docker.
- Docker receives secrets at runtime with `--env-file` or a platform secret
  store.
- Cloud Run instructions map OpenAI and service API keys from Secret Manager.
- The image was inspected separately from the running container. The running
  verification container necessarily had runtime credentials, while the image
  configuration, image history, and filesystem did not.
- Logs were searched in memory for the actual configured OpenAI key; the key
  itself was never printed.

## Authentication and production exposure

`X-API-Key` authentication protects deterministic inference and both agent
routes. `/health` and `/ready` remain public for platform probes. Authentication
is optional only in explicit local-development mode and is required by
production configuration validation.

Expected and supplied keys are compared with a constant-time comparison.
Missing and invalid keys receive the same safe `401` category. Production
OpenAPI, Swagger, and ReDoc routes are disabled.

## Rate-limit scope

The limiter is intentionally in-memory. It is suitable for the selected
single-instance portfolio deployment but is not globally consistent across
multiple processes or replicas. A distributed deployment would need a shared
store such as Redis. Cloud Run is therefore configured with a maximum of one
instance until that architecture changes.

## Error and logging safety

Known failures map to stable categories, and unknown exceptions map to a
generic `internal_error` without exception strings or stack traces in the
response. Structured logs use an allowlist and include operational metadata,
not raw request headers, API keys, full feature rows, or chain-of-thought.

One live plain-agent call returned `llm_invalid_response` after its structured
output and correction attempts were exhausted. That response was safe and
traceable by request/trace IDs; a subsequent call succeeded. This demonstrates
provider variability rather than a secret-handling failure.

## Remaining public checks

After the user deploys Cloud Run, repeat all checks against the public URL:

1. public health/readiness;
2. unauthorized and authorized prediction;
3. 429 behavior;
4. Cloud Logging field and secret inspection;
5. production docs behavior;
6. Secret Manager revision configuration.

No claim about public security has been made before those checks occur.
