# Phase 9 and 9.1 — LangGraph Agent Orchestration

## Final verified outcome

Phase 9 added a LangGraph implementation beside the existing plain-Python
router. Phase 9.1 then fixed the one unresolved compound-query case without
changing the trained model, its 0.50 threshold, the approved feature contract,
or any deterministic tool calculation.

Final measured evidence:

- plain-Python first-action routing: **40/40**
- LangGraph first-action routing: **40/40**
- expanded LangGraph full-workflow evaluation: **15/15**
- complete automated repository suite: **29 passed**
- separate `POST /agent/langgraph/query` endpoint verified
- rebuilt Docker image verified without baked secrets

The responsibility boundary remains:

> The ML model predicts.  
> Deterministic tools calculate and retrieve.  
> The LLM routes, clarifies, and writes grounded text.  
> LangGraph moves validated state between workflow steps.

## Historical progression

The evaluation history is intentionally preserved rather than overwritten:

1. The first seven-case workflow evaluation passed **2/7**.
2. Prompt and sequencing corrections improved it to **6/7**.
3. One compound order-risk plus seller-history request still incorrectly
   returned `need_clarification`.
4. Phase 9.1 diagnosed the representation problem, introduced an ordered
   action plan, expanded the suite to 15 adversarial cases, and passed
   **15/15**.

The original 6/7 result was therefore a real intermediate result, not the
current result.

## Root cause of the 6/7 result

The unresolved request asked for two distinct supported tasks:

```text
Give the risk for order <order_id> and the historical performance of seller
<seller_id>.
```

Its trace stopped here:

```text
route_request(action=need_clarification)
-> request_clarification
-> END
```

No tool had run, so the failure was not in model inference, tool execution,
graph loop-back edges, the three-call cap, or the evaluator. The original
first-decision schema could preserve only one selected tool and one current
argument. It had no structured place to retain both explicit
task-to-identifier associations, so the LLM over-applied the multiple-ID
ambiguity rule.

The generalized intent rules are now:

- two identifiers competing for one task require clarification;
- different tasks with their own identifiers form an ordered plan;
- two order IDs are valid when each is tied to a separate explicit task;
- missing, unlabeled, unknown, or unsupported identifiers require
  clarification;
- no identifier or complete query is hardcoded.

## Phase 9.1 action-plan fix

`src/agent/langgraph_agent.py` now asks the LLM for an `ActionPlan` containing
ordered `ToolAction` entries. Every entry has:

- one of the four allowlisted tool names;
- exactly the argument required by the registry;
- an `always` or `if_previous_risk_high` execution condition.

The graph stores the validated first action and a queue of pending actions. It
executes them deterministically in order rather than asking the LLM to
rediscover later tasks after each tool result. Conditional explanation remains
bounded to the explicit high-risk condition.

A narrow semantic recheck handles a provider failure mode observed during the
40-query evaluation: the LLM occasionally claimed that a query containing one
exact 32-character identifier had no identifier. The guard only triggers when
there is exactly one such token and asks the LLM to reconsider with that token
shown. It does not choose the identifier type or the tool. Zero or multiple
tokens are unchanged, so genuine ambiguity still clarifies.

Invalid JSON or schema output receives one corrective retry. Unknown tool
names cannot execute. Duplicate calls are blocked. The existing hard cap of
three tool calls prevents runaway loops and cost.

## Graph state

| Field | Meaning |
|---|---|
| `user_query` | Original natural-language request |
| `planned_actions` | Full validated ordered plan |
| `pending_actions` | Validated actions not yet executed |
| `selected_tool` | Allowlisted tool chosen for the next call |
| `tool_arguments` | Exact validated `order_id` or `seller_id` |
| `tool_results` | Authoritative outputs from deterministic tools |
| `tool_call_count` | Tools executed so far |
| `next_action` | Tool, answer, clarification, or error transition |
| `final_answer` | Natural-language text grounded in tool results |
| `clarification_message` | Specific information the user must supply |
| `error` | Safe public error data |
| `trace` | Observable node actions, outcomes, and timing |
| `status` | Running, completed, clarification, or error |
| `stop_reason` | Why execution ended |

The trained model and 96,476-row feature dataset are not copied into graph
state. They stay behind the cached deterministic tools.

## Workflow

```text
START
  |
  v
route_request
  |-- tool plan -----------> execute_tool
  |-- need clarification --> request_clarification --> END
  `-- error ---------------> handle_error ----------> END

execute_tool
  |-- success -------------> evaluate_tool_result
  `-- error ---------------> handle_error

evaluate_tool_result
  |-- pending action ------> execute_tool
  `-- complete ------------> generate_final_answer --> END
```

The exact executable registry remains:

```text
predict_delay_risk(order_id)
explain_risk(order_id)
get_seller_history(seller_id)
get_similar_past_orders(order_id)
```

There is no `eval`, dynamic import, arbitrary Python execution, model
replacement, threshold modification, or LLM-generated risk probability.

## Evaluation evidence

### Automated tests

Final suite:

```text
29 passed, 2 third-party deprecation warnings
```

Coverage includes all deterministic tools, graph prediction equality with the
saved model, high- and low-risk conditional behavior, compound task plans,
separate tasks for two order IDs, genuine ambiguity, missing and invalid
identifiers, unknown tools, the three-call cap, malformed LLM output retry,
grounding boundaries, all six LangGraph API behaviors, and the existing plain
API response contract.

The warnings come from Starlette TestClient and the Google client library under
Python 3.14; they are not failed project assertions.

### First-action routing

Both implementations passed the unchanged 40-query set:

| Implementation | Correct | Accuracy | Average latency | Median latency |
|---|---:|---:|---:|---:|
| Plain Python | 40/40 | 100% | 1,353.87 ms | 1,239.61 ms |
| LangGraph | 40/40 | 100% | 1,043.88 ms | 1,000.96 ms |

The LangGraph result is saved in
`reports/phase_9_langgraph_router_evaluation.json`. Provider/network latency
varies, so this run does not prove the framework is intrinsically faster.

### Expanded full-workflow evaluation

The Phase 9.1 set contains 15 labeled workflows and passed **15/15**. It checks:

- risk plus seller history;
- risk plus explanation;
- risk plus similar orders;
- explanation plus seller history;
- genuinely ambiguous order IDs;
- two order IDs assigned to separate tasks;
- valid order and seller IDs assigned to distinct tasks;
- missing order and seller IDs;
- unsupported actions and unknown identifier types;
- first-tool failure inside a compound plan;
- maximum-call behavior;
- one-tool and two-tool completion;
- grounded numerical output.

Average full-workflow latency was **3,074.14 ms** and median latency was
**2,618.84 ms**. The evidence is in
`reports/phase_9_langgraph_multistep_evaluation.json`.

During expansion, the grounding checker initially produced false negatives for
dates because timestamps were strings rather than numeric JSON values. The
checker was corrected to recognize numbers embedded in deterministic strings
and normalize the ISO `T` separator. Evaluation labels and agent outputs were
not weakened to obtain the final result.

### Small full-agent comparison

Across three representative queries:

| Implementation | Average | Median | Sample |
|---|---:|---:|---:|
| Plain Python | 3,250.54 ms | 3,681.17 ms | 3 |
| LangGraph | 1,888.61 ms | 1,760.63 ms | 3 |

This is a functionality check, not a reliable speed benchmark. Three external
LLM calls are too few and too sensitive to provider conditions.

## FastAPI

After every evaluation gate passed, Phase 9.1 added:

```text
POST /agent/langgraph/query
```

The endpoint is additive. The existing `POST /agent/query` plain-Python route
and Streamlit behavior remain unchanged. The LangGraph response exposes the
final answer, called tools, implementation name, trace, authoritative tool
results, clarification, safe error data, and stop reason.

API tests verify direct prediction, multi-step execution, the formerly failing
order-plus-seller case, clarification, safe missing-order handling, and the
unchanged plain-agent response contract.

## Docker and secret verification

The final API image was rebuilt and exercised:

| Measurement | Verified result |
|---|---:|
| Final image ID prefix | `sha256:a64e66` |
| Final image size | 872,320,892 bytes |
| Change from pre-9.1 image | +1,989 bytes |
| Cached rebuild time | 18.995 s |
| Fresh startup to healthy | 17.719 s |
| Baked secret variables found | 0 |
| `.env` present in image | No |

The earlier cold Phase 9 build took 437.225 seconds. The Phase 9.1 rebuild was
much faster because Docker reused the dependency layer.

Inside the rebuilt container:

- `/health` returned
  `{"status":"ok","model":"logistic_regression","phase":6}`;
- the real prediction remained exactly `0.8500189058886447`, high risk, at the
  unchanged 0.50 threshold;
- the compound LangGraph request called
  `predict_delay_risk -> get_seller_history`;
- the ambiguity case returned clarification with zero tool calls;
- OpenAPI included `/agent/langgraph/query`;
- `.env` and API keys were absent from the image.

One plain-router container request returned a transient 502 from an invalid
mixed LLM response; the retry completed normally. This existing provider-shape
instability is recorded rather than hidden and was not introduced into the
Phase 9.1 scope.

Docker endpoint latency, three requests per case:

| Case | Average | Median |
|---|---:|---:|
| Direct prediction | 55.39 ms | 31.78 ms |
| Plain agent, two tools | 7,312.68 ms | 5,767.72 ms |
| LangGraph, direct | 2,390.69 ms | 2,582.07 ms |
| LangGraph, two tools | 3,072.03 ms | 3,085.87 ms |

The final measurement required no client retries. Results are saved in
`reports/phase_9_docker_performance.json`.

## Recommendation

Keep the plain-Python router as the default UI path because it is smaller,
framework-free, and valuable evidence that the control flow was built from
first principles. Keep LangGraph as the tested alternative and expose it
through its separate API endpoint.

LangGraph becomes the stronger default only when the product needs durable
state, human approval gates, parallel branches, or long-running recovery.

## How to reproduce

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.agent.evaluate_router
.\.venv\Scripts\python.exe -m src.agent.evaluate_langgraph
.\.venv\Scripts\python.exe -m src.agent.evaluate_langgraph_multistep
.\.venv\Scripts\python.exe -m src.agent.compare_agent_performance
```

The evaluation commands make paid LLM API calls.
