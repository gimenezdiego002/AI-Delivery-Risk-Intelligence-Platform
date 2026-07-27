"""LangGraph orchestration over the existing deterministic delivery-risk tools.

The graph is additive: it does not replace ``src.agent.router`` and it does not
own model or feature logic. The LLM selects actions and writes grounded text;
the approved deterministic tools remain the only source of numerical results.
"""

from __future__ import annotations

import json
import os
import re
from time import perf_counter
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError, model_validator

from src.agent.router import (
    LLMInvalidResponseError,
    TOOL_FUNCTIONS,
    TOOL_REGISTRY,
    RouterError,
    _request_json,
)
from src.observability import emit_observation


APPROVED_TOOL_NAMES = frozenset(TOOL_FUNCTIONS)
MAX_DECISION_ATTEMPTS = 2
EXACT_IDENTIFIER_PATTERN = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32}(?![0-9a-fA-F])")


class AgentState(TypedDict):
    """Serializable state shared by every node in the graph.

    ``user_query`` is the original request. ``selected_tool`` and
    ``tool_arguments`` describe the next approved call. ``tool_results`` holds
    authoritative deterministic outputs. ``tool_call_count`` enforces the
    loop cap. Terminal nodes set ``final_answer``, ``clarification_message``,
    ``error``, and ``status``. ``trace`` contains observable actions and timing,
    never hidden model reasoning.
    """

    user_query: str
    selected_tool: str | None
    tool_arguments: dict[str, str]
    planned_actions: list[dict[str, Any]]
    pending_actions: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    tool_call_count: int
    next_action: Literal[
        "tool_call",
        "final_answer",
        "need_clarification",
        "error",
    ]
    final_answer: str | None
    clarification_message: str | None
    error: dict[str, Any] | None
    trace: list[dict[str, Any]]
    status: str
    stop_reason: str | None


class ToolAction(BaseModel):
    """One approved task with its own typed identifier and condition."""

    tool_name: Literal[
        "predict_delay_risk",
        "explain_risk",
        "get_seller_history",
        "get_similar_past_orders",
    ]
    order_id: str | None = None
    seller_id: str | None = None
    condition: Literal["always", "if_previous_risk_high"] = "always"

    @model_validator(mode="after")
    def validate_arguments(self) -> "ToolAction":
        """Require exactly the identifier declared by the selected tool."""
        _validate_tool_arguments(self.tool_name, self.order_id, self.seller_id)
        if (
            self.condition == "if_previous_risk_high"
            and self.tool_name != "explain_risk"
        ):
            raise ValueError(
                "if_previous_risk_high is supported only for explain_risk"
            )
        return self

    def arguments(self) -> dict[str, str]:
        """Return the exact validated argument dictionary."""
        return _validate_tool_arguments(
            self.tool_name, self.order_id, self.seller_id
        )


class ActionPlan(BaseModel):
    """Structured first-pass plan preserving task-to-identifier associations."""

    status: Literal["tool_plan", "need_clarification", "error"]
    actions: list[ToolAction] = Field(default_factory=list)
    clarification_message: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_status_fields(self) -> "ActionPlan":
        """Require only the fields meaningful for the selected status."""
        if self.status == "tool_plan":
            if not self.actions:
                raise ValueError("tool_plan requires at least one action")
            if self.clarification_message or self.error_message:
                raise ValueError("tool_plan cannot contain terminal text")
        elif self.status == "need_clarification":
            if not self.clarification_message:
                raise ValueError(
                    "need_clarification requires clarification_message"
                )
            if self.actions:
                raise ValueError("clarification cannot contain planned actions")
        else:
            if not self.error_message:
                raise ValueError("error status requires error_message")
            if self.actions:
                raise ValueError("error cannot contain planned actions")
        return self


class GroundedAnswer(BaseModel):
    """Strict final-answer envelope."""

    answer: str


def _registry_by_name() -> dict[str, dict[str, Any]]:
    """Return the fixed public registry indexed by approved name."""
    return {entry["name"]: entry for entry in TOOL_REGISTRY}


def _safe_error(code: str, message: str) -> dict[str, str]:
    """Create a public error without stack traces or secret values."""
    return {"code": code, "message": message}


def _public_llm_error(stage: str) -> str:
    """Return a stable public message without provider or validation internals."""
    return f"The language model returned an invalid {stage} decision."


def _validate_tool_arguments(
    tool_name: str | None,
    order_id: str | None,
    seller_id: str | None,
) -> dict[str, str]:
    """Validate a call against the fixed registry's exact argument contract."""
    registry = _registry_by_name()
    if tool_name not in APPROVED_TOOL_NAMES or tool_name not in registry:
        raise ValueError(f"Unknown or unapproved tool: {tool_name}")

    available = {"order_id": order_id, "seller_id": seller_id}
    arguments = {key: value for key, value in available.items() if value is not None}
    required = set(registry[tool_name]["required_arguments"])
    if set(arguments) != required:
        raise ValueError(
            f"Tool '{tool_name}' requires exactly {sorted(required)}; "
            f"received {sorted(arguments)}."
        )
    if any(not value.strip() for value in arguments.values()):
        raise ValueError("Tool identifiers cannot be blank.")
    return arguments


def _parse_plan(raw_response: str) -> ActionPlan:
    """Parse strict JSON and validate the complete ordered action plan."""
    payload = json.loads(raw_response)
    if not isinstance(payload, dict):
        raise ValueError("Action plan must be a JSON object.")
    allowed_keys = set(ActionPlan.model_fields)
    unknown_keys = set(payload).difference(allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown action-plan keys: {sorted(unknown_keys)}")
    return ActionPlan.model_validate(payload)


def _request_plan(prompt: str) -> ActionPlan:
    """Request one validated plan, with one controlled corrective retry."""
    correction = ""
    last_error: Exception | None = None
    for attempt in range(MAX_DECISION_ATTEMPTS):
        raw_response = _request_json(
            prompt + correction,
            ActionPlan,
            max_output_tokens=1_500,
        )
        try:
            return _parse_plan(raw_response)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            correction = (
                "\nThe previous response was invalid because: "
                f"{exc}. Return corrected JSON only."
            )
            if attempt == MAX_DECISION_ATTEMPTS - 1:
                break
    raise LLMInvalidResponseError(
        "The LLM failed to return a valid LangGraph plan after one retry: "
        f"{last_error}"
    )


def _routing_prompt(user_query: str) -> str:
    """Build the ordered planning prompt from the fixed registry."""
    return f"""You plan delivery-risk requests using deterministic tools.

Available tool registry:
{json.dumps(TOOL_REGISTRY, indent=2)}

Routing rules:
- Return one ordered action for every explicit, supported task in the query.
- Copy identifiers exactly; never invent, shorten, or alter them.
- Risk/probability/will-it-be-late questions use predict_delay_risk.
- Why/explanation/risk-driver/signal/feature-contribution questions use
  explain_risk.
- Seller performance/history/reliability questions use get_seller_history.
- Similar/comparable/past-order questions use get_similar_past_orders.
- Associate each identifier with the task that directly names or describes it.
- Two IDs competing for one task, such as "is order A or order B late", are
  ambiguous and require need_clarification.
- Multiple IDs are not ambiguous when each has a separate explicit task, such
  as "predict order A and explain order B".
- An order task plus a seller task is an ordered multi-task request, not an
  ambiguity. Preserve the user's left-to-right task order.
- Each action must contain only its own required identifier.
- A conditional request such as "if risk is high, explain it" uses
  condition="if_previous_risk_high" on the explanation action.
- All other actions use condition="always".
- If a requested tool lacks its required identifier, ask for that specific ID.
- If identifier type or task association is unclear, return need_clarification.
- Phrases such as "my order", "this order", "the order", or "it" are not
  identifiers unless a prior action in the same plan supplies the exact ID for
  an explicit same-order follow-up.
- If the request is unsupported, return need_clarification and explain the
  supported delivery-risk capabilities.
- Never calculate or invent risk values, seller statistics, feature
  contributions, similar orders, or tool outputs.
- Return only JSON matching the supplied schema.

User query:
{user_query}
"""


def decide_langgraph_action(user_query: str) -> dict[str, Any]:
    """Return the first action plus the remaining validated action queue."""
    if not user_query.strip():
        return {
            "status": "need_clarification",
            "tool_name": None,
            "arguments": {},
            "planned_actions": [],
            "pending_actions": [],
            "clarification_message": (
                "What delivery-risk question can I help with?"
            ),
        }
    prompt = _routing_prompt(user_query)
    plan = _request_plan(prompt)
    exact_identifiers = EXACT_IDENTIFIER_PATTERN.findall(user_query)
    if plan.status == "need_clarification" and len(exact_identifiers) == 1:
        plan = _request_plan(
            prompt
            + "\nValidation guard: the query contains exactly one explicit "
            f"32-character identifier: {exact_identifiers[0]}. Re-evaluate the "
            "request using that exact token. Do not infer its type from format; "
            "associate it only from the user's order/seller wording and task. "
            "Keep need_clarification if the task is unsupported or the type or "
            "intent remains unclear."
        )
    if plan.status != "tool_plan":
        return {
            "status": plan.status,
            "tool_name": None,
            "arguments": {},
            "planned_actions": [],
            "pending_actions": [],
            "clarification_message": plan.clarification_message,
            "error_message": plan.error_message,
        }

    serialized_actions = [
        action.model_dump(mode="json") for action in plan.actions
    ]
    first_action = plan.actions[0]
    return {
        "status": "tool_call",
        "tool_name": first_action.tool_name,
        "arguments": first_action.arguments(),
        "planned_actions": serialized_actions,
        "pending_actions": serialized_actions[1:],
        "clarification_message": None,
        "error_message": None,
    }


def _grounded_answer_prompt(state: AgentState) -> str:
    """Build the cap/duplicate completion prompt from authoritative results."""
    return f"""Answer the delivery-risk query using only the deterministic tool
results below. Do not add facts, infer causes, calculate values, or invent
outputs. If information is unavailable, say so. Preserve the
correlational-not-causal caveat when an explanation is present.

User query:
{state["user_query"]}

Tool results:
{json.dumps(state["tool_results"], indent=2)}

Write concise plain-language prose, not a nested JSON object or data dump.
Return JSON with exactly one string field named "answer".
"""


def _request_grounded_answer(state: AgentState) -> str:
    """Generate a validated answer after a graph safeguard forces completion."""
    correction = ""
    last_error: Exception | None = None
    for attempt in range(MAX_DECISION_ATTEMPTS):
        raw_response = _request_json(
            _grounded_answer_prompt(state) + correction,
            GroundedAnswer,
            max_output_tokens=1_000,
        )
        try:
            payload = json.loads(raw_response)
            if not isinstance(payload, dict) or set(payload) != {"answer"}:
                raise ValueError("Grounded answer must contain only 'answer'.")
            return GroundedAnswer.model_validate(payload).answer
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            correction = (
                f"\nPrevious response invalid: {exc}. Return corrected JSON only."
            )
            if attempt == MAX_DECISION_ATTEMPTS - 1:
                break
    raise LLMInvalidResponseError(
        "The LLM failed to return a grounded answer after one retry: "
        f"{last_error}"
    )


def route_request(state: AgentState) -> dict[str, Any]:
    """Select the first tool or terminal clarification action."""
    started = perf_counter()
    try:
        action = decide_langgraph_action(state["user_query"])
        event = {
            "node": "route_request",
            "action": action["status"],
            "selected_tool": action.get("tool_name"),
            "arguments": action.get("arguments", {}),
            "planned_actions": action.get("planned_actions", []),
            "elapsed_ms": round((perf_counter() - started) * 1_000, 3),
        }
        return {
            "selected_tool": action.get("tool_name"),
            "tool_arguments": action.get("arguments", {}),
            "planned_actions": action.get("planned_actions", []),
            "pending_actions": action.get("pending_actions", []),
            "next_action": action["status"],
            "clarification_message": action.get("clarification_message"),
            "error": (
                _safe_error(
                    "planning_error",
                    action.get("error_message")
                    or "The request could not be planned safely.",
                )
                if action["status"] == "error"
                else state["error"]
            ),
            "trace": [*state["trace"], event],
        }
    except Exception:
        return {
            "next_action": "error",
            "error": _safe_error(
                "routing_error", _public_llm_error("routing")
            ),
            "trace": [
                *state["trace"],
                {
                    "node": "route_request",
                    "action": "error",
                    "elapsed_ms": round((perf_counter() - started) * 1_000, 3),
                },
            ],
        }


def execute_tool(state: AgentState) -> dict[str, Any]:
    """Validate and execute exactly one allowlisted deterministic tool."""
    started = perf_counter()
    try:
        tool_name = state["selected_tool"]
        arguments = _validate_tool_arguments(
            tool_name,
            state["tool_arguments"].get("order_id"),
            state["tool_arguments"].get("seller_id"),
        )
        signature = (tool_name, tuple(sorted(arguments.items())))
        prior_signatures = {
            (
                event.get("selected_tool"),
                tuple(sorted(event.get("arguments", {}).items())),
            )
            for event in state["trace"]
            if event.get("node") == "execute_tool"
        }
        if signature in prior_signatures:
            return {
                "next_action": "final_answer",
                "stop_reason": "duplicate_call_prevented",
                "trace": [
                    *state["trace"],
                    {
                        "node": "execute_tool",
                        "action": "duplicate_call_prevented",
                        "selected_tool": tool_name,
                        "arguments": arguments,
                        "elapsed_ms": round(
                            (perf_counter() - started) * 1_000, 3
                        ),
                    },
                ],
            }

        result = TOOL_FUNCTIONS[tool_name](**arguments)
        tool_call_count = state["tool_call_count"] + 1
        elapsed_ms = round((perf_counter() - started) * 1_000, 3)
        emit_observation(
            "deterministic_tool_call",
            selected_tool=tool_name,
            latency_ms=elapsed_ms,
            outcome_ok=bool(result.get("ok", False)),
            agent_implementation="langgraph",
        )
        event = {
            "node": "execute_tool",
            "action": "tool_result",
            "selected_tool": tool_name,
            "arguments": arguments,
            "outcome_ok": bool(result.get("ok", False)),
            "elapsed_ms": elapsed_ms,
        }
        if not result.get("ok", False):
            return {
                "tool_results": [
                    *state["tool_results"],
                    {"tool_name": tool_name, "arguments": arguments, "result": result},
                ],
                "tool_call_count": tool_call_count,
                "next_action": "error",
                "error": result.get(
                    "error",
                    _safe_error("tool_error", "The deterministic tool failed."),
                ),
                "trace": [*state["trace"], event],
            }
        return {
            "tool_results": [
                *state["tool_results"],
                {"tool_name": tool_name, "arguments": arguments, "result": result},
            ],
            "tool_call_count": tool_call_count,
            "next_action": "final_answer",
            "trace": [*state["trace"], event],
        }
    except Exception:
        return {
            "next_action": "error",
            "error": _safe_error(
                "tool_execution_error",
                "The approved deterministic tool could not be executed.",
            ),
            "trace": [
                *state["trace"],
                {
                    "node": "execute_tool",
                    "action": "error",
                    "elapsed_ms": round((perf_counter() - started) * 1_000, 3),
                },
            ],
        }


def _condition_is_met(
    action: ToolAction, tool_results: list[dict[str, Any]]
) -> bool:
    """Evaluate the small allowlisted set of deterministic action conditions."""
    if action.condition == "always":
        return True
    for item in reversed(tool_results):
        if item.get("tool_name") == "predict_delay_risk":
            return item.get("result", {}).get("risk_level") == "high"
    return False


def evaluate_tool_result(state: AgentState) -> dict[str, Any]:
    """Advance through the validated action queue or finish safely."""
    started = perf_counter()
    max_tool_calls = _max_tool_calls()
    if state["tool_call_count"] >= max_tool_calls:
        return {
            "next_action": "final_answer",
            "stop_reason": "max_tool_calls_reached",
            "trace": [
                *state["trace"],
                {
                    "node": "evaluate_tool_result",
                    "action": "final_answer",
                    "reason": "max_tool_calls_reached",
                    "elapsed_ms": round((perf_counter() - started) * 1_000, 3),
                },
            ],
        }

    remaining = list(state["pending_actions"])
    trace = list(state["trace"])
    while remaining:
        action_payload = remaining.pop(0)
        try:
            action = ToolAction.model_validate(action_payload)
        except ValidationError:
            return {
                "next_action": "error",
                "error": _safe_error(
                    "invalid_pending_action",
                    "A planned action failed registry validation.",
                ),
                "trace": [
                    *trace,
                    {
                        "node": "evaluate_tool_result",
                        "action": "error",
                        "elapsed_ms": round(
                            (perf_counter() - started) * 1_000, 3
                        ),
                    },
                ],
            }

        if not _condition_is_met(action, state["tool_results"]):
            trace.append(
                {
                    "node": "evaluate_tool_result",
                    "action": "condition_not_met",
                    "selected_tool": action.tool_name,
                    "condition": action.condition,
                    "elapsed_ms": round(
                        (perf_counter() - started) * 1_000, 3
                    ),
                }
            )
            continue

        arguments = action.arguments()
        return {
            "selected_tool": action.tool_name,
            "tool_arguments": arguments,
            "pending_actions": remaining,
            "next_action": "tool_call",
            "trace": [
                *trace,
                {
                    "node": "evaluate_tool_result",
                    "action": "tool_call",
                    "selected_tool": action.tool_name,
                    "arguments": arguments,
                    "condition": action.condition,
                    "elapsed_ms": round(
                        (perf_counter() - started) * 1_000, 3
                    ),
                },
            ],
        }

    return {
        "pending_actions": [],
        "next_action": "final_answer",
        "trace": [
            *trace,
            {
                "node": "evaluate_tool_result",
                "action": "final_answer",
                "reason": "planned_actions_complete",
                "elapsed_ms": round((perf_counter() - started) * 1_000, 3),
            },
        ],
    }


def generate_final_answer(state: AgentState) -> dict[str, Any]:
    """Finish with grounded text from the loop decision or recorded results."""
    started = perf_counter()
    try:
        answer = state["final_answer"] or _request_grounded_answer(state)
        return {
            "final_answer": answer,
            "status": "completed",
            "next_action": "final_answer",
            "stop_reason": state["stop_reason"] or "enough_information",
            "trace": [
                *state["trace"],
                {
                    "node": "generate_final_answer",
                    "action": "completed",
                    "elapsed_ms": round((perf_counter() - started) * 1_000, 3),
                },
            ],
        }
    except Exception:
        return {
            "next_action": "error",
            "error": _safe_error(
                "answer_generation_error",
                "The language model could not produce a grounded final answer.",
            ),
            "trace": [
                *state["trace"],
                {
                    "node": "generate_final_answer",
                    "action": "error",
                    "elapsed_ms": round((perf_counter() - started) * 1_000, 3),
                },
            ],
        }


def request_clarification(state: AgentState) -> dict[str, Any]:
    """Terminate safely with a specific request for missing information."""
    return {
        "status": "need_clarification",
        "final_answer": state["clarification_message"]
        or "Please provide the required order or seller identifier.",
        "stop_reason": "clarification_required",
        "trace": [
            *state["trace"],
            {
                "node": "request_clarification",
                "action": "need_clarification",
                "elapsed_ms": 0.0,
            },
        ],
    }


def handle_error(state: AgentState) -> dict[str, Any]:
    """Terminate with a safe structured error and no internal stack trace."""
    error = state["error"] or _safe_error(
        "agent_error", "The LangGraph workflow could not complete the request."
    )
    return {
        "status": "error",
        "final_answer": error["message"],
        "error": error,
        "stop_reason": "error",
        "trace": [
            *state["trace"],
            {"node": "handle_error", "action": "error", "elapsed_ms": 0.0},
        ],
    }


def _max_tool_calls() -> int:
    """Load and validate the same three-call default as the original router."""
    load_dotenv(override=False)
    value = int(os.getenv("LLM_MAX_TOOL_CALLS", "3"))
    if value < 1:
        raise RouterError("LLM_MAX_TOOL_CALLS must be at least 1.")
    return value


def _after_route(state: AgentState) -> str:
    """Map the first decision to its next node."""
    return {
        "tool_call": "execute_tool",
        "need_clarification": "request_clarification",
        "final_answer": "generate_final_answer",
        "error": "handle_error",
    }[state["next_action"]]


def _after_execution(state: AgentState) -> str:
    """Send successful tool results to evaluation and failures to error."""
    if state["next_action"] == "error":
        return "handle_error"
    if state["stop_reason"] == "duplicate_call_prevented":
        return "generate_final_answer"
    return "evaluate_tool_result"


def _after_evaluation(state: AgentState) -> str:
    """Route loop-back, terminal completion, clarification, or error."""
    return {
        "tool_call": "execute_tool",
        "final_answer": "generate_final_answer",
        "need_clarification": "request_clarification",
        "error": "handle_error",
    }[state["next_action"]]


def _after_final_answer(state: AgentState) -> str:
    """Recover safely if answer generation itself failed."""
    return "handle_error" if state["next_action"] == "error" else END


def build_langgraph():
    """Compile the explicit Phase 9 workflow."""
    graph = StateGraph(AgentState)
    graph.add_node("route_request", route_request)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("evaluate_tool_result", evaluate_tool_result)
    graph.add_node("generate_final_answer", generate_final_answer)
    graph.add_node("request_clarification", request_clarification)
    graph.add_node("handle_error", handle_error)

    graph.add_edge(START, "route_request")
    graph.add_conditional_edges(
        "route_request",
        _after_route,
        {
            "execute_tool": "execute_tool",
            "request_clarification": "request_clarification",
            "generate_final_answer": "generate_final_answer",
            "handle_error": "handle_error",
        },
    )
    graph.add_conditional_edges(
        "execute_tool",
        _after_execution,
        {
            "evaluate_tool_result": "evaluate_tool_result",
            "generate_final_answer": "generate_final_answer",
            "handle_error": "handle_error",
        },
    )
    graph.add_conditional_edges(
        "evaluate_tool_result",
        _after_evaluation,
        {
            "execute_tool": "execute_tool",
            "generate_final_answer": "generate_final_answer",
            "request_clarification": "request_clarification",
            "handle_error": "handle_error",
        },
    )
    graph.add_conditional_edges(
        "generate_final_answer",
        _after_final_answer,
        {"handle_error": "handle_error", END: END},
    )
    graph.add_edge("request_clarification", END)
    graph.add_edge("handle_error", END)
    return graph.compile(name="delivery_risk_langgraph_agent")


LANGGRAPH_APP = build_langgraph()


def _initial_state(user_query: str) -> AgentState:
    """Create a complete, JSON-serializable graph state."""
    return {
        "user_query": user_query,
        "selected_tool": None,
        "tool_arguments": {},
        "planned_actions": [],
        "pending_actions": [],
        "tool_results": [],
        "tool_call_count": 0,
        "next_action": "need_clarification",
        "final_answer": None,
        "clarification_message": None,
        "error": None,
        "trace": [],
        "status": "running",
        "stop_reason": None,
    }


def run_langgraph_agent(user_query: str) -> dict[str, Any]:
    """Run the compiled graph and return a stable, inspectable response."""
    final_state = LANGGRAPH_APP.invoke(_initial_state(user_query))
    return {
        "ok": final_state["status"] != "error",
        "status": final_state["status"],
        "answer": final_state["final_answer"] or "",
        "tool_call_count": final_state["tool_call_count"],
        "tool_results": final_state["tool_results"],
        "clarification_message": final_state["clarification_message"],
        "error": final_state["error"],
        "stop_reason": final_state["stop_reason"],
        "trace": final_state["trace"],
        "implementation": "langgraph",
    }
