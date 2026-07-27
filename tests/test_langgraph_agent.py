"""Focused tests for the additive LangGraph orchestration layer."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.agent.langgraph_agent import (
    APPROVED_TOOL_NAMES,
    _validate_tool_arguments,
    run_langgraph_agent,
)
from src.agent.tools import predict_delay_risk


ORDER_ID = "be55f985440dddd650b389a55db8e49c"
SECOND_ORDER_ID = "9694aa09499321709cdb542840ebbbb2"
LOW_RISK_ORDER_ID = "6340164ffcc87a11dd0ad37d2551994c"
SELLER_ID = "3078096983cf766a32a06257648502d1"


def _action(
    tool_name: str,
    identifier: str,
    condition: str = "always",
) -> dict[str, str]:
    field = "seller_id" if tool_name == "get_seller_history" else "order_id"
    return {
        "tool_name": tool_name,
        field: identifier,
        "condition": condition,
    }


def _tool_plan(*actions: dict[str, str]) -> str:
    return json.dumps({"status": "tool_plan", "actions": list(actions)})


def _grounded_answer(answer: str = "Grounded answer.") -> str:
    return json.dumps({"answer": answer})


def _run_direct_tool(tool_name: str, identifier: str) -> dict:
    with patch(
        "src.agent.langgraph_agent._request_json",
        side_effect=[
            _tool_plan(_action(tool_name, identifier)),
            _grounded_answer(),
        ],
    ):
        return run_langgraph_agent(
            f"Use {tool_name} for {'seller' if 'seller' in tool_name else 'order'} "
            f"{identifier}."
        )


def test_direct_risk_prediction_matches_deterministic_tool() -> None:
    """The graph must return the exact saved-model prediction from the tool."""
    direct = predict_delay_risk(ORDER_ID)
    result = _run_direct_tool("predict_delay_risk", ORDER_ID)

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert result["tool_call_count"] == 1
    graph_prediction = result["tool_results"][0]["result"]
    assert graph_prediction == direct
    assert graph_prediction["late_delivery_probability"] == pytest.approx(
        direct["late_delivery_probability"]
    )


@pytest.mark.parametrize(
    ("tool_name", "identifier"),
    [
        ("explain_risk", ORDER_ID),
        ("get_seller_history", SELLER_ID),
        ("get_similar_past_orders", ORDER_ID),
    ],
)
def test_direct_tool_routes_complete(tool_name: str, identifier: str) -> None:
    """Each remaining approved deterministic tool executes through the graph."""
    result = _run_direct_tool(tool_name, identifier)
    assert result["ok"] is True
    assert result["status"] == "completed"
    assert result["tool_results"][0]["tool_name"] == tool_name
    assert result["tool_results"][0]["result"]["ok"] is True


def test_high_risk_conditional_plan_executes_explanation() -> None:
    """A high-risk condition should advance to the planned explanation."""
    with patch(
        "src.agent.langgraph_agent._request_json",
        side_effect=[
            _tool_plan(
                _action("predict_delay_risk", ORDER_ID),
                _action(
                    "explain_risk",
                    ORDER_ID,
                    condition="if_previous_risk_high",
                ),
            ),
            _grounded_answer("The real tools report high risk and its signals."),
        ],
    ):
        result = run_langgraph_agent(
            f"Predict risk for order {ORDER_ID} and if high explain why."
        )

    assert result["status"] == "completed"
    assert [item["tool_name"] for item in result["tool_results"]] == [
        "predict_delay_risk",
        "explain_risk",
    ]
    assert result["tool_results"][0]["result"]["risk_level"] == "high"


def test_low_risk_conditional_plan_skips_explanation() -> None:
    """A false high-risk condition must skip, not execute, the second action."""
    direct = predict_delay_risk(LOW_RISK_ORDER_ID)
    assert direct["risk_level"] == "low"
    with patch(
        "src.agent.langgraph_agent._request_json",
        side_effect=[
            _tool_plan(
                _action("predict_delay_risk", LOW_RISK_ORDER_ID),
                _action(
                    "explain_risk",
                    LOW_RISK_ORDER_ID,
                    condition="if_previous_risk_high",
                ),
            ),
            _grounded_answer(),
        ],
    ):
        result = run_langgraph_agent(
            f"Predict order {LOW_RISK_ORDER_ID}; explain only if high."
        )

    assert [item["tool_name"] for item in result["tool_results"]] == [
        "predict_delay_risk"
    ]
    assert any(
        event.get("action") == "condition_not_met"
        for event in result["trace"]
    )


def test_order_plus_seller_plan_preserves_associations() -> None:
    """Distinct typed identifiers should execute their distinct requested tasks."""
    with patch(
        "src.agent.langgraph_agent._request_json",
        side_effect=[
            _tool_plan(
                _action("predict_delay_risk", ORDER_ID),
                _action("get_seller_history", SELLER_ID),
            ),
            _grounded_answer(),
        ],
    ):
        result = run_langgraph_agent(
            f"Give risk for order {ORDER_ID} and history for seller {SELLER_ID}."
        )

    assert [item["tool_name"] for item in result["tool_results"]] == [
        "predict_delay_risk",
        "get_seller_history",
    ]
    assert result["tool_results"][0]["arguments"] == {"order_id": ORDER_ID}
    assert result["tool_results"][1]["arguments"] == {"seller_id": SELLER_ID}


def test_two_order_ids_with_separate_tasks_are_not_ambiguous() -> None:
    """Each explicitly associated order task should remain in the queue."""
    with patch(
        "src.agent.langgraph_agent._request_json",
        side_effect=[
            _tool_plan(
                _action("predict_delay_risk", ORDER_ID),
                _action("explain_risk", SECOND_ORDER_ID),
            ),
            _grounded_answer(),
        ],
    ):
        result = run_langgraph_agent(
            f"Predict order {ORDER_ID} and explain order {SECOND_ORDER_ID}."
        )

    assert [
        (item["tool_name"], item["arguments"])
        for item in result["tool_results"]
    ] == [
        ("predict_delay_risk", {"order_id": ORDER_ID}),
        ("explain_risk", {"order_id": SECOND_ORDER_ID}),
    ]


def test_multiple_ambiguous_order_ids_request_clarification() -> None:
    """Two IDs competing for one task must not be guessed."""
    clarification = json.dumps(
        {
            "status": "need_clarification",
            "clarification_message": "Which order ID should I use?",
        }
    )
    with patch(
        "src.agent.langgraph_agent._request_json",
        return_value=clarification,
    ):
        result = run_langgraph_agent("Is order 123 or order 456 late?")

    assert result["status"] == "need_clarification"
    assert result["tool_call_count"] == 0
    assert result["tool_results"] == []


def test_two_exact_ambiguous_ids_do_not_trigger_single_id_guard() -> None:
    """The semantic recheck must not weaken genuine multiple-ID ambiguity."""
    clarification = json.dumps(
        {
            "status": "need_clarification",
            "clarification_message": "Which order ID should I use?",
        }
    )
    with patch(
        "src.agent.langgraph_agent._request_json",
        return_value=clarification,
    ) as request:
        result = run_langgraph_agent(
            f"Is order {ORDER_ID} or order {SECOND_ORDER_ID} late?"
        )
    assert result["status"] == "need_clarification"
    assert request.call_count == 1


def test_single_exact_id_guard_rechecks_false_missing_id_clarification() -> None:
    """One detected token may prompt re-evaluation but never local tool choice."""
    clarification = json.dumps(
        {
            "status": "need_clarification",
            "clarification_message": "Please provide the order ID.",
        }
    )
    with patch(
        "src.agent.langgraph_agent._request_json",
        side_effect=[
            clarification,
            _tool_plan(_action("explain_risk", ORDER_ID)),
            _grounded_answer(),
        ],
    ) as request:
        result = run_langgraph_agent(
            f"What are the strongest risk drivers for order {ORDER_ID}?"
        )
    assert result["status"] == "completed"
    assert result["tool_results"][0]["tool_name"] == "explain_risk"
    assert request.call_count == 3


def test_unknown_identifier_type_requests_clarification() -> None:
    """Unlabeled values must not be assigned to tools by guesswork."""
    clarification = json.dumps(
        {
            "status": "need_clarification",
            "clarification_message": "What are A and B, and what should I check?",
        }
    )
    with patch(
        "src.agent.langgraph_agent._request_json",
        return_value=clarification,
    ):
        result = run_langgraph_agent("Check A and B.")
    assert result["status"] == "need_clarification"
    assert result["tool_call_count"] == 0


def test_missing_order_returns_safe_structured_error() -> None:
    """A missing order must terminate safely without an invented response."""
    missing_id = "definitely-not-a-real-order"
    with patch(
        "src.agent.langgraph_agent._request_json",
        return_value=_tool_plan(
            _action("predict_delay_risk", missing_id)
        ),
    ):
        result = run_langgraph_agent(f"Predict risk for order {missing_id}.")

    assert result["ok"] is False
    assert result["status"] == "error"
    assert result["error"]["code"] == "order_not_found"
    assert result["tool_results"][0]["result"]["ok"] is False


def test_invalid_seller_returns_safe_structured_error() -> None:
    """An invalid seller ID should preserve the deterministic tool error."""
    missing_id = "definitely-not-a-real-seller"
    with patch(
        "src.agent.langgraph_agent._request_json",
        return_value=_tool_plan(
            _action("get_seller_history", missing_id)
        ),
    ):
        result = run_langgraph_agent(f"Show seller {missing_id} history.")

    assert result["status"] == "error"
    assert result["error"]["code"] == "seller_not_found"


def test_unknown_tool_is_rejected() -> None:
    """The graph must never expose arbitrary Python execution."""
    with pytest.raises(ValueError, match="Unknown or unapproved tool"):
        _validate_tool_arguments("os.system", None, None)
    assert APPROVED_TOOL_NAMES == {
        "predict_delay_risk",
        "explain_risk",
        "get_seller_history",
        "get_similar_past_orders",
    }


def test_maximum_tool_call_protection() -> None:
    """The configured cap must stop a longer valid plan after one tool."""
    with (
        patch.dict("os.environ", {"LLM_MAX_TOOL_CALLS": "1"}),
        patch(
            "src.agent.langgraph_agent._request_json",
            side_effect=[
                _tool_plan(
                    _action("predict_delay_risk", ORDER_ID),
                    _action("explain_risk", ORDER_ID),
                ),
                _grounded_answer("Answer grounded in the one tool result."),
            ],
        ),
    ):
        result = run_langgraph_agent(
            f"Predict and explain order {ORDER_ID}."
        )

    assert result["status"] == "completed"
    assert result["tool_call_count"] == 1
    assert result["stop_reason"] == "max_tool_calls_reached"


def test_invalid_llm_output_retries_once_then_fails_safely() -> None:
    """Two malformed plans should become a structured graph error."""
    with patch(
        "src.agent.langgraph_agent._request_json",
        side_effect=["not-json", '{"status": "tool_plan"}'],
    ) as request:
        result = run_langgraph_agent(f"Predict risk for order {ORDER_ID}.")

    assert request.call_count == 2
    assert result["status"] == "error"
    assert result["error"]["code"] == "routing_error"


def test_initial_llm_answer_cannot_bypass_deterministic_prediction() -> None:
    """A fake first-step numerical answer is outside the accepted plan schema."""
    fake_answer = json.dumps(
        {
            "status": "final_answer",
            "final_answer": "The probability is 99%.",
        }
    )
    with patch(
        "src.agent.langgraph_agent._request_json",
        return_value=fake_answer,
    ):
        result = run_langgraph_agent(f"Predict risk for order {ORDER_ID}.")

    assert result["status"] == "error"
    assert result["tool_call_count"] == 0
    assert result["tool_results"] == []
    assert "99%" not in result["answer"]
