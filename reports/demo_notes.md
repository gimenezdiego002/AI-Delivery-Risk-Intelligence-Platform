# Phase 7 Demo Notes

## How to run the full demo

Open two terminals from the project root.

Terminal 1 — start the FastAPI service:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

Confirm the API is online:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok","model":"logistic_regression","phase":6}
```

Terminal 2 — start the Streamlit UI:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/app/streamlit_app.py
```

Then open the Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

The Streamlit app calls the FastAPI endpoints over HTTP. It does not import the
model, tools, or router directly.

## Useful demo IDs

### High-risk order

```text
be55f985440dddd650b389a55db8e49c
```

Why it is useful:

- Produces a high risk score around 85%.
- Explanation shows strong contributions from seller-customer distance, seller
  state, product category, and category historical late rate.
- Works well for the “Order Risk,” “Risk Explanation,” and loop-back agent demos.

### Low-risk order

```text
6340164ffcc87a11dd0ad37d2551994c
```

Why it is useful:

- Produces a low risk score around 15%.
- Good contrast against the high-risk example.

### Seller history example

```text
3078096983cf766a32a06257648502d1
```

Why it is useful:

- Returns a non-empty seller history snapshot.
- Shows historical volume, late rate, review score, and the leakage-safe history rule.

## Agent demo queries

### Single-tool query

```text
Will order 6340164ffcc87a11dd0ad37d2551994c arrive late?
```

Expected behavior:

- Calls `predict_delay_risk`.
- Returns a low-risk answer.

### Loop-back query

```text
First predict delay risk for order be55f985440dddd650b389a55db8e49c and if it is high, explain why.
```

Expected behavior:

- First calls `predict_delay_risk`.
- Because the order is high risk and the user asked why, it then calls `explain_risk`.
- The trace should show both tools in order.

### Clarification query

```text
Will my order be late?
```

Expected behavior:

- Does not call a tool.
- Asks the user to provide an exact order ID.

## Screenshot assets

Screenshots are saved in:

```text
reports/demo_screenshots/
```

Current screenshots:

- `order_risk_high.png` — high-risk order prediction.
- `risk_explanation.png` — feature-contribution explanation.
- `agent_loopback_trace.png` — loop-back agent answer with trace expanded.

## Known limitation

The current system predicts and explains orders that already exist in the
processed historical feature dataset. A truly new live order would need a
production feature builder that creates the same 23 model features at request
time.
