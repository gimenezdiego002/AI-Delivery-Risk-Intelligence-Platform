# Phase 10 Deployment Platform Decision

Decision date: 2026-07-27

## Workload facts

- Final optimized image: 205,317,857 bytes
- Loaded local container memory after exercised prediction/agent traffic:
  approximately 329–347 MiB (329.4 MiB in the final capture)
- Prepared inference table: 54,336,910 bytes
- External HTTPS access to OpenAI is required for agent routes
- Model/data are immutable and bundled; no persistent disk is required
- The API needs runtime secrets, logs, health probes, and scale control
- Agent requests can take several seconds, while deterministic prediction is
  much faster

## Comparison

| Platform | Container and secrets | Cost model | Sleep/cold start | Memory fit | Operational fit |
|---|---|---|---|---|---|
| Google Cloud Run | Builds the repository Dockerfile, provides Secret Manager integration, HTTPS, environment settings, probes, and Cloud Logging | Pay per request/resource with a monthly free tier; billing account is still required | Scales to zero by default; first request starts an instance | Configurable; 1 GiB gives useful headroom | Strongest controls, observability, and portfolio deployment story |
| Render | Direct Dockerfile/Git deploy, environment secrets, health checks, logs, TLS, custom domains | Free web-service preview or paid fixed instance | Free service sleeps after 15 minutes; documented spin-up is about one minute | Free and Starter are 512 MB, risky for the measured 347 MiB baseline plus similarity work | Simplest UI, but free resource margin is too small |
| Railway | Detects a root Dockerfile, variables, health check, logs, domains | Free plan includes $1 monthly credit; Hobby is $5/month including $5 usage, then usage-priced | Usage/sleep behavior is configurable but costs depend on consumed RAM/CPU | Free is 0.5 GB; the same headroom concern applies | Excellent developer experience, but less predictable cost/memory fit here |
| Fly.io | Docker/Machine model, secrets, health checks, regional placement | Pay-as-you-go; new users receive a limited trial, not a permanent general free tier | Machines can stop/start; behavior is more infrastructure-oriented | 1 GiB shared machine is suitable but paid | Flexible, but more operational surface than this portfolio API needs |

## Verified current-source facts

- Render documents 512 MB and 0.1 CPU for Free, and its free service sleeps
  after 15 idle minutes:
  <https://render.com/docs/compute-plans> and
  <https://render.com/docs/free>.
- Render supports Docker services, environment secrets, health checks, TLS,
  logs, and custom domains:
  <https://render.com/docs/docker>,
  <https://render.com/docs/configure-environment-variables>, and
  <https://render.com/docs/health-checks>.
- Railway documents Free at $0 with $1 monthly resource credit, Hobby at
  $5/month with $5 included usage, a 0.5 GB Free memory limit, Dockerfile
  detection, and deployment health checks:
  <https://docs.railway.com/pricing/plans>,
  <https://docs.railway.com/builds/dockerfiles>, and
  <https://docs.railway.com/deployments/healthchecks>.
- Cloud Run supports source deployment using an existing Dockerfile, managed
  HTTPS, scale to zero, configurable memory, Secret Manager, health probes,
  request timeouts, and structured stdout logs:
  <https://docs.cloud.google.com/run/docs/deploying-source-code>,
  <https://docs.cloud.google.com/run/docs/overview/what-is-cloud-run>,
  <https://docs.cloud.google.com/run/docs/configuring/services/memory-limits>,
  <https://docs.cloud.google.com/run/docs/configuring/services/secrets>,
  <https://docs.cloud.google.com/run/docs/configuring/healthchecks>, and
  <https://docs.cloud.google.com/run/docs/logging>.
- Fly.io documents pay-as-you-go Machine pricing and states that it has a
  limited free trial rather than a general ongoing free account:
  <https://fly.io/docs/about/pricing/> and
  <https://fly.io/docs/about/free-trial/>.

Pricing and limits change. Recheck each linked official page before entering
billing information.

## Selection: Google Cloud Run

Cloud Run is selected because the 1 GiB configuration gives safer memory
headroom than the 512 MB free offerings, while request-based billing and scale
to zero fit an interview/demo API. Secret Manager, structured Cloud Logging,
startup/liveness probes, managed HTTPS, and explicit instance caps align
directly with Phase 10.

Initial deployment settings:

- 1 vCPU
- 1 GiB memory
- request-based billing
- minimum instances: 0
- maximum instances: 1
- concurrency: 4
- request timeout: 120 seconds
- container port: 8000
- startup/readiness path: `/ready`
- liveness path: `/health`
- public Cloud Run ingress with application-level `X-API-Key`

`max-instances=1` is deliberate for the initial deployment because the
in-memory rate limiter is not globally consistent across replicas. It limits
horizontal scalability and is not equivalent to a distributed rate limiter.

## Why deployment is not executed automatically

Cloud Run requires:

- a user-owned Google Cloud project;
- billing enabled;
- authenticated `gcloud` access;
- Cloud Run, Cloud Build, Artifact Registry, and Secret Manager APIs;
- permission to create services, builds, and secrets;
- the user's production OpenAI and backend API keys.

None of those are safely inferable or authorizable from the repository.
Repository preparation is complete, but the public deployment remains pending
until the user completes the account/billing step and explicitly authorizes
deployment.
