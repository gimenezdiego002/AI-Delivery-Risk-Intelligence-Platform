# Phase 5 — Plain-Python LLM Router Explanation

## What Phase 5 adds

Phase 5 adds a natural-language routing layer on top of the deterministic Phase
4 tools. The important idea is that the LLM does **not** predict delivery risk
itself. The LLM only decides which already-tested Python tool should be called.

The actual business logic still lives in `src/agent/tools.py`:

- `predict_delay_risk(order_id)`
- `explain_risk(order_id)`
- `get_seller_history(seller_id)`
- `get_similar_past_orders(order_id)`

The router lives in `src/agent/router.py`.

## Why this matters

This separates the project into two clean responsibilities:

1. The machine learning model and deterministic tools produce grounded results.
2. The LLM interprets the user's natural-language request and chooses the right
   tool.

That is safer and more professional than asking an LLM to invent predictions
directly.

## Security checks before using OpenAI

Before using the OpenAI API, the project was checked for common secret-leak
risks:

- `.env` is ignored by git.
- `.env.example` is tracked, but it only contains placeholder values.
- `.vscode/settings.json` is ignored by git.
- `mlflow.db` is ignored by git.
- Source files were scanned for OpenAI/Gemini key patterns.
- No real OpenAI or Gemini API key was found in tracked source files.
- The router loads API keys from environment variables and does not print them.
- Generated router checkpoint output is ignored with:

```gitignore
reports/phase_5_router_predictions.csv
```

Important: the API key should stay only in `.env`. Never paste it into Python
files, notebooks, README files, screenshots, or commits.

## OpenAI configuration

The project now supports both OpenAI and Gemini through `.env`.

Recommended Phase 5 setup:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your-real-openai-api-key
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0
LLM_MAX_TOOL_CALLS=3
LLM_EVAL_DELAY_SECONDS=1.0
```

Gemini is still supported as an alternative, but OpenAI was used here because
the Gemini free tier hit quota/rate limits during the 40-query router
evaluation.

## Files created or changed

- `src/agent/router.py`
  - Added OpenAI support.
  - Kept Gemini support.
  - Uses strict JSON responses and validates them with Pydantic.
  - Adds a hard cap of 3 tool calls to prevent infinite loops.

- `src/agent/evaluate_router.py`
  - Evaluates first-tool selection on the labeled query set.
  - Reports overall accuracy, per-tool accuracy, and a confusion matrix.
  - Handles provider rate-limit/server retry cases.

- `tests/agent_eval_queries.csv`
  - Contains 40 labeled natural-language queries.
  - Covers all four tools plus clarification cases.

- `.env.example`
  - Documents the OpenAI/Gemini environment variables safely.

- `requirements.txt`
  - Adds `openai` and `pytest`.

- `.gitignore`
  - Ignores generated Phase 5 router checkpoint predictions.

## Router evaluation result

The Phase 5 router evaluation ran on all 40 labeled queries using OpenAI.

Result:

```text
Correct: 40/40
Tool-selection accuracy: 100.00%
```

Per-tool breakdown:

| Expected tool | Correct | Total | Accuracy |
|---|---:|---:|---:|
| `predict_delay_risk` | 8 | 8 | 100% |
| `explain_risk` | 8 | 8 | 100% |
| `get_seller_history` | 8 | 8 | 100% |
| `get_similar_past_orders` | 8 | 8 | 100% |
| `need_clarification` | 8 | 8 | 100% |

The router originally got 39/40 because the ambiguous query “Will my order be
late?” produced an invalid mixed response. The router prompt was tightened so
phrases like “my order” or “this order” are treated as missing identifiers
unless an exact order ID is provided.

## Clarification example

User query:

```text
Will my order be late?
```

Router result:

```text
status: need_clarification
answer: Please provide the exact order identifier.
tool_call_count: 0
```

This is correct because the user did not provide an `order_id`.

## Loop-back example

User query:

```text
First predict delay risk for order be55f985440dddd650b389a55db8e49c and if it is high, explain why.
```

The agent performed two tool calls:

1. `predict_delay_risk(order_id)`
2. `explain_risk(order_id)`

The first tool found:

```text
late_delivery_probability: 0.8500
risk_level: high
```

Because the risk was high and the user asked for an explanation, the agent
looped back and called `explain_risk`.

The final answer referenced only values from the tool outputs, including:

- high late-delivery risk
- probability around 0.85
- seller-customer distance
- seller state
- product category
- category historical late rate
- the correlational-not-causal caveat

## What this phase proves

Phase 5 proves that the project can translate natural language into reliable
tool use. That is the start of the “agent” part of the portfolio project.

The strongest resume-ready idea from this phase is:

> Built a plain-Python LLM routing layer over four deterministic delivery-risk
> tools, achieving 100% tool-selection accuracy on 40 labeled natural-language
> queries while keeping model predictions grounded in tested scikit-learn
> outputs.

## What Phase 6 should be

Phase 6 should expose the working agent through FastAPI.

Recommended endpoints:

- `POST /agent/query`
  - Input: natural-language user query.
  - Output: final answer, tool trace, and status.

- `POST /orders/{order_id}/risk`
  - Directly calls `predict_delay_risk`.

- `POST /orders/{order_id}/explain`
  - Directly calls `explain_risk`.

- `GET /sellers/{seller_id}/history`
  - Directly calls `get_seller_history`.

- `GET /orders/{order_id}/similar`
  - Directly calls `get_similar_past_orders`.

FastAPI should come after Phase 5 because the Python agent loop is now working
and measured. The API should expose this existing logic, not rewrite it.
