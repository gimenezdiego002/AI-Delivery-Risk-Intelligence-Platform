# Phase 9.1 — Plain Python versus LangGraph

## Executive result

Both orchestration implementations achieved **40/40** correct first-action
selections on the same labeled query set. After the Phase 9.1 action-plan fix,
LangGraph also passed **15/15** adversarial full-workflow cases, including the
compound order-risk plus seller-history query that had previously stopped the
seven-case set at 6/7.

LangGraph did not improve measured routing accuracy because the plain router
was already perfect. Its value is explicit state, ordered plans, named nodes,
conditional edges, and node-level observability. The plain implementation
remains the simpler default for the existing Streamlit path.

## Measured comparison

| Category | Plain Python | LangGraph |
|---|---:|---:|
| First-action accuracy | 40/40 (100%) | 40/40 (100%) |
| Average first-action latency | 1,353.87 ms | 1,043.88 ms |
| Median first-action latency | 1,239.61 ms | 1,000.96 ms |
| Representative full-agent average | 3,250.54 ms | 1,888.61 ms |
| Representative full-agent median | 3,681.17 ms | 1,760.63 ms |
| Full-agent sample size | 3 | 3 |
| Saved adversarial workflow set | None | 15/15 |
| Default maximum tool calls | 3 | 3 |
| Direct agent-framework dependency | None | `langgraph` |
| API endpoint | `POST /agent/query` | `POST /agent/langgraph/query` |

These timings are provider- and network-sensitive. The three-query full-agent
sample is a functionality comparison, not proof that LangGraph is inherently
faster. The 40-query routing result is stronger evidence for correctness than
for performance.

Measured sources:

- `reports/phase_9_plain_router_latency_checkpoint.json`
- `reports/phase_9_langgraph_router_evaluation.json`
- `reports/phase_9_agent_comparison_metrics.json`
- `reports/phase_9_langgraph_multistep_evaluation.json`
- `reports/phase_9_docker_performance.json`

## Behavioral comparison

| Category | Plain Python | LangGraph |
|---|---|---|
| Multi-step control | Visible `while` loop asks for the next step after each result | Validated ordered plan advances through graph state |
| Clarification | Direct return from `run_agent()` | Named terminal clarification node |
| Loop safeguards | Counter plus duplicate-call set | State counter, duplicate signatures, conditional terminal edges |
| Testability | Direct functions; existing router evaluation | Node/LLM boundaries plus dedicated graph and API tests |
| Trace | Decision and tool events | Plan, node names, arguments, result status, timing, stop reason |
| Complexity | Lower | Higher, with an extra framework and state concepts |
| Best fit | Small bounded synchronous workflow | More branching, durable execution, approval, or recovery |

## Why the earlier 6/7 result happened

The progression was **2/7 -> 6/7 -> 15/15**.

Prompt corrections first resolved unsolicited follow-up tools and sequencing.
The final 6/7 failure was structural: the old first-decision schema could hold
only one tool and one identifier, so the LLM lost the second explicit
task-to-ID association and treated a valid compound request as ambiguous.

Phase 9.1 replaced that single choice with an ordered `ActionPlan`. Every
`ToolAction` is independently validated against the fixed registry, stored in
state, and executed in written order. Two IDs competing for one action still
clarify; separate task/ID pairs do not.

The 15-case evaluation now includes compound tasks, two IDs tied to separate
tasks, genuine ambiguity, missing identifiers, unsupported requests, unknown
identifier types, first-tool errors, call-cap behavior, and grounded one- and
two-tool completions.

## Safety and grounding

Both implementations call only:

```text
predict_delay_risk(order_id)
explain_risk(order_id)
get_seller_history(seller_id)
get_similar_past_orders(order_id)
```

The LLM cannot replace the saved model, calculate a probability, change the
threshold, execute arbitrary code, or fabricate seller statistics. Final
language must be grounded in separately returned deterministic `tool_results`.

The LangGraph planner gets one JSON/schema retry. A narrow one-ID semantic
recheck corrects the observed case where the provider falsely said an exact
single identifier was missing, without selecting the tool or weakening
multiple-ID ambiguity.

## API and deployment

The existing `/agent/query` endpoint and Streamlit UI still use the plain
router. LangGraph is available through the separate
`/agent/langgraph/query` endpoint, so users can compare implementations without
silently changing existing behavior.

The final Docker image:

- rebuilt successfully at 872,320,892 bytes;
- grew only 1,989 bytes from the pre-9.1 image;
- started healthy in 17.719 seconds;
- produced the identical saved-model prediction;
- completed the fixed compound LangGraph sequence;
- contained neither `.env` nor an API-key environment variable.

## Recommendation

Keep both:

- default to plain Python for the current portfolio demo because its control
  flow is smaller and easy to explain from first principles;
- use the separate LangGraph route to demonstrate explicit orchestration,
  multi-step state, validation, and observability.

Switch the default only when workflow complexity—durable state, approvals,
parallel work, or recovery—materially benefits from the framework.
