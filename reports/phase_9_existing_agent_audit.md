# Phase 9 Existing Agent Audit

## Scope and verified inputs

This audit was completed before any LangGraph implementation was added. It
documents the behavior that exists in the current plain-Python agent and maps
each responsibility to the planned LangGraph equivalent.

Files inspected:

- `src/agent/router.py`
- `src/agent/tools.py`
- `src/agent/evaluate_router.py`
- `tests/test_agent_tools.py`
- `tests/agent_eval_queries.csv`
- `src/api/main.py`
- `requirements.txt`
- `reports/phase_5_explicit.md`
- `reports/phase_5_router_evaluation.json`

Verified facts:

- The existing agent is a hand-written `while` loop, not an agent framework.
- The only executable tools are the four callables in `TOOL_FUNCTIONS`.
- The saved Phase 5 evaluation reports 40/40 correct first-tool selections
  using `gpt-4o-mini`.
- There is no dedicated unit-test file for router orchestration. The existing
  automated tests cover the deterministic tools, while
  `src/agent/evaluate_router.py` evaluates first-tool routing.
- The existing FastAPI endpoint calls `run_agent()` and returns a reduced
  response containing the final answer and called tool names.

## Current flow and future LangGraph mapping

| Current behavior | Plain-Python implementation | LangGraph equivalent |
|---|---|---|
| 1. Query entry | `run_agent(user_query)` receives a string. FastAPI passes `AgentQueryRequest.query` to it. | `run_langgraph_agent(user_query)` initializes typed graph state and invokes the compiled graph. |
| 2. First action selection | `decide_action()` sends the query and `TOOL_REGISTRY` to the configured LLM. | A `route_request` node performs the same validated first decision. |
| 3. Structured validation | `ActionDecision`, `LoopDecision`, and `FinalResponse` are Pydantic models. JSON is parsed locally and unknown keys are rejected. | LangGraph nodes reuse equivalent Pydantic decision schemas; graph state is separately typed. |
| 4. Argument validation | `_validate_registry_arguments()` requires the exact argument names declared by the selected registry entry and rejects missing, extra, or blank identifiers. | The routing and execution nodes validate against an allowlisted tool specification before changing state or executing anything. |
| 5. Tool execution | `TOOL_FUNCTIONS[tool_name](**arguments)` calls one of four imported deterministic functions. | An `execute_tool` node uses the same fixed mapping; no `eval`, dynamic import, or arbitrary lookup is allowed. |
| 6. Result storage | A `tool_result` event containing the tool name, arguments, and structured result is appended to `trace`. | The result is appended to serializable `tool_results` and an observable trace event is appended to graph state. |
| 7. Loop-back decision | `_decide_next_step()` receives the original query, full trace, and registry. It returns another tool call, a final answer, or clarification. | An `evaluate_tool_result` node sets the next action; conditional edges route back to `execute_tool` or onward to a terminal node. |
| 8. Clarification | Missing or ambiguous identifiers return `need_clarification`. `run_agent()` exits with the clarification question and zero or more prior tool calls. | Conditional routing sends state to `request_clarification`, which produces a terminal, structured response. |
| 9. Final answer | A valid loop decision may contain a grounded `final_answer`. `_grounded_final_answer()` is used when the call cap or duplicate-call protection forces completion. | `generate_final_answer` produces a grounded answer from recorded tool results only. |
| 10. Maximum calls | `LLM_MAX_TOOL_CALLS` defaults to 3. The loop checks the count immediately after every tool call. | State contains `tool_call_count`; a conditional edge prevents another execution when the configured cap is reached. |
| 11. Errors | Deterministic tools return `{"ok": false, "error": {"code": ..., "message": ...}}`. Router/schema/provider failures raise `RouterError`. FastAPI converts repeated router failures to HTTP 502. | Tool failures remain structured state. A `handle_error` terminal node returns safe messages without stack traces or secrets. |
| 12. Trace | The returned trace contains `decision` and `tool_result` events. It exposes actions and results, not hidden reasoning. | Graph state keeps observable node transitions, selected tools, arguments, outcomes, and elapsed time; no chain-of-thought is stored or returned. |

## Existing structured schemas

### `ActionDecision`

- `status`: `tool_call` or `need_clarification`
- `tool_name`: one of the four approved names, or `null`
- `order_id`: string or `null`
- `seller_id`: string or `null`
- `clarification_question`: string or `null`

Its model validator requires a tool name for a tool call, forbids clarification
text on a tool call, and forbids tool fields when clarification is requested.

### `LoopDecision`

- `status`: `tool_call`, `final_answer`, or `need_clarification`
- the same optional tool and identifier fields
- `final_answer`
- `clarification_question`

Its model validator requires the content appropriate to the selected status.
Final-answer decisions tolerate irrelevant nullable tool fields but still
require non-empty answer text.

### `FinalResponse`

- `answer`: string

This is used when duplicate-call protection or the maximum-call cap requires a
grounded completion.

## Approved deterministic tool boundary

The exact executable registry is:

1. `predict_delay_risk(order_id)`
2. `explain_risk(order_id)`
3. `get_seller_history(seller_id)`
4. `get_similar_past_orders(order_id)`

The LLM selects a name and copies an identifier. It does not receive authority
to run arbitrary Python. Model predictions, feature contributions, seller
statistics, and neighbor results come only from these deterministic functions.

## Provider and secret handling

The router supports OpenAI and Gemini. It loads `.env` through
`python-dotenv`, then reads provider, model, temperature, call cap, and API keys
from environment variables. Keys are not embedded in prompts, traces, or source
code. Client objects are cached in-process.

The OpenAI path uses the existing `openai` SDK with JSON Schema response
formatting. The Gemini path uses `google-genai` with a Pydantic response schema.
Both responses are parsed and validated locally.

## Retry, duplicate, and cost safeguards

- One initial LLM response plus one corrective retry is allowed for malformed
  or schema-invalid output.
- An identical tool name plus argument set cannot execute twice.
- Tool calls stop at `LLM_MAX_TOOL_CALLS`, which defaults to 3.
- The first-decision evaluator includes provider-capacity retries and an
  optional delay between paid requests.

## Existing evidence and gaps

Existing evidence:

- `tests/test_agent_tools.py` has five passing-intent tests for deterministic
  tool behavior, leakage boundaries, feature-contract fidelity, and neighbor
  exclusion.
- `tests/agent_eval_queries.csv` contains 40 labeled queries: eight for each
  tool and eight clarification cases.
- `reports/phase_5_router_evaluation.json` records 40/40 correct selections and
  zero router errors using `gpt-4o-mini`.

Gaps Phase 9 must address:

- There is no dedicated deterministic orchestration test suite for the
  plain-Python router.
- The Phase 5 evaluation records accuracy but not per-query latency.
- The plain-Python trace does not explicitly record node names or elapsed time.
- Full multi-step behavior does not have a saved labeled evaluation artifact.

These gaps do not invalidate the working Phase 5 result. They define the
comparison and observability work required for Phase 9.

## Design constraint for the LangGraph implementation

LangGraph will orchestrate the same responsibilities through explicit state,
nodes, and conditional edges. It will not own the model, dataset, risk
calculation, explanation mathematics, seller aggregation, or nearest-neighbor
logic. The original router remains the supported reference implementation and
must continue to pass its existing regression checks.
