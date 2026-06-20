# Phase 4 Deterministic Tool Report

## Scope

Phase 4 wraps the existing saved Logistic Regression pipeline and feature
contract. It does not retrain a model, rebuild processed data, route natural
language, or call an LLM API.

## Tools

### `predict_delay_risk(order_id)`

Validates all 23 saved feature names and values, then returns the saved model's
late-delivery probability, high/low risk level, model name, and threshold.
Unknown orders and incomplete rows produce structured errors.

### `explain_risk(order_id)`

Multiplies the order's transformed feature values by the saved Logistic
Regression coefficients, aggregates contributions to original feature names,
and ranks them by absolute log-odds magnitude. The summary is a deterministic
template. Contributions are correlations within the model, not causal effects.

### `get_seller_history(seller_id, as_of_order_id=None)`

Returns Phase 2's precomputed seller volume, late rate, and review average at a
specific order-time cutoff. Reusing those columns preserves the exact original
leakage rule: later orders, unresolved deliveries, and not-yet-created reviews
cannot enter the snapshot.

### `get_similar_past_orders(order_id, top_n=5)`

Fits a local nearest-neighbor search after median imputation/scaling of numeric
features and one-hot encoding of categories. Only orders delivered before the
query purchase time are candidates, and an explicit self-match check prevents
the query order from appearing in results.

## Reviewed real examples

| Order | Predicted risk | Level | Actual outcome | Notable association |
|---|---:|---|---|---|
| `6340164ffcc87a11dd0ad37d2551994c` | 0.150 | low | on time | longer estimated window reduced modeled risk |
| `a8ae22d68419abf4bbe92b7623b08657` | 0.500 | high | on time | customer/seller state increased modeled risk |
| `be55f985440dddd650b389a55db8e49c` | 0.850 | high | on time | 2,271 km distance increased modeled risk |

All four tools returned successful structured output for all three orders. The
two high-risk false positives are consistent with the baseline's known low
precision and should not be hidden.

## Tests

Five deterministic tests verify:

1. A valid order returns a probability strictly between zero and one.
2. An unknown order returns `order_not_found` instead of crashing.
3. A mid-history seller snapshot excludes known later orders.
4. Similar orders exclude the query and were delivered before query time.
5. Explanations only contain names from `models/model_features.json`.

Run with:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agent_tools -v
```

## Limitation

The current tools retrieve prepared historical order rows. A truly new live
order needs a production feature builder that creates the same 23 features as
of the prediction timestamp.

## Phase 5 recommendation

Create roughly 40 labeled natural-language queries with an `expected_tool`
field, implement a narrow tool router, and measure exact tool-selection
accuracy. Keep the four deterministic functions as the source of business
logic; routing should only decide which function to call.
