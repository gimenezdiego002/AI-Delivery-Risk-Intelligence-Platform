# Project Phase Status

This table reflects verified repository evidence after Phase 10 hardening.

| Phase | Status | Main evidence | What it proves |
|---|---|---|---|
| 1 — Data foundation | Complete and verified | `src/data/`, processed dataset, Phase 1 report | Olist tables produce a reusable order-level dataset |
| 2 — Feature engineering | Complete and verified | `delivery_features.csv`, leakage audit | Distance and historical features respect prediction time |
| 3 — Model comparison | Complete and verified | saved model, feature contract, MLflow reports | Three classifiers were compared on future orders |
| 4 — Deterministic tools | Complete and verified | `src/agent/tools.py`, tool tests | Four authoritative tools work without an LLM |
| 5 — Plain router | Complete and verified | fresh 40-query Phase 10 live artifact | Plain first-action routing measured 40/40 |
| 6 — FastAPI | Complete and verified | API source, API tests, performance reports | Deterministic tools and both agents are exposed through HTTP |
| 7 — Streamlit | Complete and verified | UI source, screenshots, demo notes | Separate HTTP client provides a portfolio demo |
| 8 — Docker | Complete and verified | Dockerfile and container evidence | FastAPI runs reproducibly without baked secrets |
| 9/9.1 — LangGraph | Complete and verified | 40/40 routing, 15/15 workflow report, endpoint and Docker evidence | Ordered graph orchestration handles compound requests and preserves deterministic boundaries |
| 10 — Production hardening | Mostly complete | typed settings, JSON logs, auth/rate tests, 205 MB image, CI/deploy configs | Repository implementation and local verification are complete; public Cloud Run deployment, hosted CI, monitoring, and public latency require user-owned access |

## Key verified numbers

- Processed dataset: **96,476 orders**
- Selected model: **Logistic Regression**
- Selected-model F1: **0.1666**
- Selected-model recall: **0.8400**
- Fresh Phase 10 plain-Python routing: **40/40**
- Fresh Phase 10 LangGraph routing: **40/40**
- Fresh Phase 10 LangGraph full workflows: **15/15**
- Automated repository tests: **63 passed**
- Final Docker image: **205,317,857 bytes**
- Image reduction from Phase 9 baseline: **76.46%**
- Verified probability: **0.8500189058886447**
- Secrets baked into image: **0 found**

## Current architecture

```text
Olist data -> leakage-safe features -> saved Logistic Regression
                                      |
                                      v
                           four deterministic tools
                             /                 \
                            v                   v
                  plain-Python router     LangGraph action plan
                            |                   |
                            v                   v
                    /agent/query       /agent/langgraph/query
                             \                 /
                              v               v
                          authenticated FastAPI
                           /health + /ready
                                  |
                     local Streamlit HTTP client
                                  |
                      optimized non-root Docker
                                  |
                    Cloud Run configuration ready
```

The Streamlit demo remains on the default plain-router endpoint. LangGraph is
additive and independently accessible.

## Phase 10 deployment boundary

Cloud Run is the selected backend platform. No public deployment has been
claimed. The remaining owner actions are:

1. choose or create a billed Google Cloud project;
2. authenticate `gcloud` and enable the required APIs;
3. add OpenAI and backend API keys to Secret Manager;
4. run the documented Cloud Run deployment;
5. execute public authentication, prediction parity, rate-limit, cold/warm
   latency, log, and uptime checks;
6. push the branch so GitHub-hosted deterministic CI actually runs.

Until these are completed, Phase 10 remains **Mostly complete**, not fully
complete. Phase 11 should be a deployment-validation milestone rather than a
new model, RAG, or agent-framework phase.
