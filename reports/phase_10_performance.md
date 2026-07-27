# Phase 10 Performance Report

## Scope and caution

Local and local-Docker measurements were executed on July 27, 2026. Each
endpoint has only three samples, so p95 is the maximum under the script's
nearest-rank definition and should not be treated as a stable production
percentile. Provider/network variation dominates agent timings.

Public cold and warm rows are intentionally unmeasured because no public Cloud
Run deployment exists.

## Average latency

| Environment | Health | Prediction | Plain Agent* | LangGraph Direct | LangGraph Two-Tool |
|---|---:|---:|---:|---:|---:|
| Local | 21.98 ms | 379.41 ms | 4,112.57 ms | 2,795.29 ms | 3,658.79 ms |
| Docker local | 20.80 ms | 46.18 ms | 2,997.69 ms | 2,606.59 ms | 3,548.29 ms |
| Public deployment cold | Not measured | Not measured | Not measured | Not measured | Not measured |
| Public deployment warm | Not measured | Not measured | Not measured | Not measured | Not measured |

\*The benchmark's plain query was “Predict and explain …”. The plain router
returned a successful clarification response rather than executing two tools,
so this number is an agent-request latency, **not a plain two-tool latency**.
A separate established conditional wording successfully called
`predict_delay_risk` then `explain_risk`.

## Median and p95

| Environment / case | Count | Median | p95 | Errors |
|---|---:|---:|---:|---:|
| Local health | 3 | 18.86 ms | 30.59 ms | 0 |
| Local prediction | 3 | 18.36 ms | 1,104.58 ms | 0 |
| Local plain agent | 3 | 4,240.75 ms | 4,744.78 ms | 0 |
| Local LangGraph direct | 3 | 2,902.67 ms | 3,065.08 ms | 0 |
| Local LangGraph two-tool | 3 | 3,697.66 ms | 3,998.54 ms | 0 |
| Docker health | 3 | 28.05 ms | 28.47 ms | 0 |
| Docker prediction | 3 | 48.21 ms | 52.51 ms | 0 |
| Docker plain agent | 3 | 2,949.95 ms | 3,663.12 ms | 0 |
| Docker LangGraph direct | 3 | 2,539.73 ms | 2,746.95 ms | 0 |
| Docker LangGraph two-tool | 3 | 3,598.96 ms | 3,623.80 ms | 0 |

Full average, median, p95, minimum, maximum, error counts, and zero client
retries are saved in:

- `reports/phase_10_performance_local.json`
- `reports/phase_10_performance_docker_local.json`

## Docker measurements

| Metric | Before | Final Phase 10 | Change |
|---|---:|---:|---:|
| Image size | 872,320,892 bytes | 205,317,857 bytes | -667,003,035 bytes (-76.46%) |
| Cold build | Not re-run for old image | 82.259 s initial optimized build | Measured separately |
| Cached final build | N/A | 6.048 s | dependency layers reused |
| Startup to readiness | N/A | 5.273 s | local Docker Desktop |

The final image is approximately 195.8 MiB. The reduction comes primarily from
installing runtime-only dependencies and excluding Streamlit, notebooks,
MLflow, XGBoost, visualization packages, raw data, reports, and development
tools.

## Interpretation

Docker did not cause a meaningful latency regression in this sample. The local
prediction average is misleading because one first-load request took 1.10
seconds while its median was only 18.36 ms. Agent differences are within normal
external-provider variation at this sample size.

The Phase 10 live LangGraph routing evaluation (40 calls) measured 1,181.72 ms
average and 1,092.90 ms median for the first decision. The 15-case full workflow
evaluation measured 3,182.77 ms average and 2,971.90 ms median.

## Public measurements still required

After deployment:

1. time the first request after scale-to-zero;
2. run at least 20 warm deterministic requests;
3. run a cost-controlled agent sample;
4. record Cloud Run region, instance memory, retries, and errors;
5. add public cold/warm rows without replacing the local evidence.
