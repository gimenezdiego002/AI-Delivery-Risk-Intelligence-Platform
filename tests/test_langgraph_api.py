"""FastAPI tests for the additive LangGraph endpoint and plain endpoint."""

from __future__ import annotations

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app


ORDER_ID = "be55f985440dddd650b389a55db8e49c"
SELLER_ID = "3078096983cf766a32a06257648502d1"
client = TestClient(app)


def _action(tool_name: str, identifier: str) -> dict[str, str]:
    field = "seller_id" if tool_name == "get_seller_history" else "order_id"
    return {"tool_name": tool_name, field: identifier, "condition": "always"}


def _plan(*actions: dict[str, str]) -> str:
    return json.dumps({"status": "tool_plan", "actions": list(actions)})


def _answer(text: str = "Grounded tool answer.") -> str:
    return json.dumps({"answer": text})


def _post_langgraph(query: str, responses: list[str]):
    with patch(
        "src.agent.langgraph_agent._request_json",
        side_effect=responses,
    ):
        return client.post("/agent/langgraph/query", json={"query": query})


def test_langgraph_direct_prediction_endpoint() -> None:
    """A direct request should return one authoritative tool result."""
    response = _post_langgraph(
        f"Predict order {ORDER_ID}.",
        [_plan(_action("predict_delay_risk", ORDER_ID)), _answer()],
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["implementation"] == "langgraph"
    assert payload["tools_called"] == ["predict_delay_risk"]
    assert 0 < payload["tool_results"][0]["result"][
        "late_delivery_probability"
    ] < 1


def test_langgraph_multistep_loop_endpoint() -> None:
    """A two-action plan should expose both observable tool calls."""
    response = _post_langgraph(
        f"Predict and explain order {ORDER_ID}.",
        [
            _plan(
                _action("predict_delay_risk", ORDER_ID),
                _action("explain_risk", ORDER_ID),
            ),
            _answer(),
        ],
    )
    payload = response.json()
    assert payload["tools_called"] == [
        "predict_delay_risk",
        "explain_risk",
    ]
    assert payload["tool_call_count"] == 2


def test_langgraph_order_plus_seller_endpoint() -> None:
    """The fixed compound request should preserve typed associations."""
    response = _post_langgraph(
        f"Risk for order {ORDER_ID} and history for seller {SELLER_ID}.",
        [
            _plan(
                _action("predict_delay_risk", ORDER_ID),
                _action("get_seller_history", SELLER_ID),
            ),
            _answer(),
        ],
    )
    payload = response.json()
    assert payload["tools_called"] == [
        "predict_delay_risk",
        "get_seller_history",
    ]
    assert payload["status"] == "completed"


def test_langgraph_clarification_endpoint() -> None:
    """Genuinely ambiguous IDs should return clarification without tools."""
    response = _post_langgraph(
        "Is order A or order B late?",
        [
            json.dumps(
                {
                    "status": "need_clarification",
                    "clarification_message": "Which order should I check?",
                }
            )
        ],
    )
    payload = response.json()
    assert payload["status"] == "need_clarification"
    assert payload["tools_called"] == []
    assert payload["clarification_message"] == "Which order should I check?"


def test_langgraph_missing_order_error_endpoint() -> None:
    """A nonexistent order should remain a safe structured tool error."""
    missing_id = "ffffffffffffffffffffffffffffffff"
    response = _post_langgraph(
        f"Predict order {missing_id}.",
        [_plan(_action("predict_delay_risk", missing_id))],
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "order_not_found"


def test_existing_plain_agent_endpoint_regression() -> None:
    """The original endpoint contract must remain unchanged."""
    plain_result = {
        "ok": True,
        "status": "completed",
        "answer": "Existing plain router answer.",
        "tool_call_count": 1,
        "trace": [
            {
                "event": "tool_result",
                "tool_name": "predict_delay_risk",
            }
        ],
    }
    with patch("src.api.main.run_agent", return_value=plain_result):
        response = client.post("/agent/query", json={"query": "test query"})
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "status": "completed",
        "answer": "Existing plain router answer.",
        "tools_called": ["predict_delay_risk"],
        "tool_call_count": 1,
    }
