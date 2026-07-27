"""Streamlit demo UI for the Delivery Risk Intelligence Agent.

The UI talks to the FastAPI service over HTTP. It deliberately does not import
model, tool, or router modules so the presentation layer stays separate from
the project logic layer.
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("API_REQUEST_TIMEOUT_SECONDS", "30"))


def _load_backend_api_key() -> str | None:
    """Read the backend key server-side without placing it in UI source."""
    environment_key = os.getenv("API_KEY")
    if environment_key:
        return environment_key
    try:
        return st.secrets.get("API_KEY")
    except StreamlitSecretNotFoundError:
        return None


BACKEND_API_KEY = _load_backend_api_key()


def call_api(method: str, path: str, **kwargs: Any) -> tuple[dict[str, Any] | None, str | None, int | None]:
    """Call the FastAPI service and return parsed JSON, plain error, and status."""
    headers = dict(kwargs.pop("headers", {}))
    if BACKEND_API_KEY and path not in {"/health", "/ready"}:
        headers.setdefault("X-API-Key", BACKEND_API_KEY)
    try:
        response = requests.request(
            method,
            f"{API_BASE_URL}{path}",
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
    except requests.RequestException as exc:
        return None, f"Could not reach the API service: {exc}", None

    try:
        payload = response.json()
    except ValueError:
        return None, "The API returned a non-JSON response.", response.status_code

    if response.status_code >= 400:
        message = payload.get("error", {}).get("message", "The API returned an error.")
        return payload, message, response.status_code
    return payload, None, response.status_code


def render_sidebar() -> None:
    """Render the compact project control panel and API health indicator."""
    st.sidebar.title("Delivery Risk Intel")
    st.sidebar.caption("Predict late-delivery risk and explain operational signals.")

    payload, error, status_code = call_api("GET", "/health")
    st.sidebar.divider()
    st.sidebar.subheader("Service Status")
    if payload and status_code == 200 and payload.get("status") == "ok":
        st.sidebar.success("API online")
        st.sidebar.caption(f"Model: {payload.get('model')} · Phase {payload.get('phase')}")
    else:
        st.sidebar.error("API offline")
        st.sidebar.caption(error or "Start FastAPI before using the demo.")


def apply_design() -> None:
    """Apply a focused logistics/ops visual system."""
    st.markdown(
        """
        <style>
        :root {
            --risk-red: #E53E3E;
            --risk-green: #38A169;
            /* Steel blue feels operational/logistics-oriented: serious,
               dashboard-friendly, and less generic than Streamlit blue. */
            --ops-accent: #2B6CB0;
            --panel: #111827;
            --panel-soft: #1F2937;
            --text-muted: #A0AEC0;
        }
        .stApp {
            background: #0B1120;
            color: #EDF2F7;
        }
        section[data-testid="stSidebar"] {
            background: #0F172A;
            border-right: 1px solid #253047;
        }
        .block-container {
            padding-top: 2rem;
        }
        div[data-testid="stTabs"] button {
            color: #CBD5E0;
        }
        .risk-card, .summary-box, .answer-box, .mini-card {
            background: var(--panel);
            border: 1px solid #263449;
            border-radius: 16px;
            padding: 1.25rem;
            box-shadow: 0 12px 30px rgba(0,0,0,0.25);
        }
        .risk-card {
            border-left: 6px solid var(--ops-accent);
        }
        .risk-label {
            font-size: 4rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            line-height: 1;
        }
        .metric-line {
            margin-top: 0.8rem;
            font-size: 1.15rem;
            color: #E2E8F0;
        }
        .mini-label {
            color: var(--text-muted);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .mini-value {
            font-size: 2.2rem;
            font-weight: 800;
        }
        .summary-box {
            margin-top: 1rem;
            color: #E2E8F0;
            line-height: 1.55;
        }
        .answer-box {
            border-left: 6px solid var(--ops-accent);
            font-size: 1.1rem;
            line-height: 1.6;
        }
        .footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #253047;
            color: var(--text-muted);
            text-align: center;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render portfolio metrics as a compact footer."""
    st.markdown(
        """
        <div class="footer">
        Built on Olist Brazilian E-Commerce dataset · Logistic Regression ·
        96,476 orders · 40/40 agent routing accuracy
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_plain_error(error: str | None, status_code: int | None) -> None:
    """Show API errors as plain English instead of raw JSON."""
    if status_code == 404:
        st.error(error or "That record was not found.")
    elif status_code == 422:
        st.warning(error or "This record is missing required model features.")
    else:
        st.error(error or "Something went wrong while calling the API.")


def render_order_risk_tab() -> None:
    """Render the order-level late-delivery risk prediction tab."""
    st.subheader("Order Risk")
    st.write("Enter an order ID to retrieve the saved model's risk prediction.")
    demo_order_id = (
        st.query_params.get("order_id")
        if st.query_params.get("demo") == "risk"
        else None
    )

    with st.form("risk_form"):
        order_id = st.text_input(
            "Order ID",
            placeholder="6340164ffcc87a11dd0ad37d2551994c",
            value=demo_order_id or "",
        ).strip()
        submitted = st.form_submit_button("Predict risk")

    if demo_order_id:
        submitted = True
    if not submitted:
        return
    if not order_id:
        st.warning("Please enter an order ID.")
        return

    payload, error, status_code = call_api("GET", f"/orders/{order_id}/risk")
    if error:
        render_plain_error(error, status_code)
        return

    risk_level = str(payload["risk_level"]).upper()
    color = "#E53E3E" if risk_level == "HIGH" else "#38A169"
    probability = float(payload["late_delivery_probability"]) * 100
    st.markdown(
        f"""
        <div class="risk-card">
          <div class="risk-label" style="color:{color};">{risk_level}</div>
          <div class="metric-line">Late-delivery probability: <b>{probability:.2f}%</b></div>
          <div class="metric-line">Model: <b>{payload["model_name"]}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_explanation_tab() -> None:
    """Render deterministic feature-contribution explanations."""
    st.subheader("Risk Explanation")
    st.write("See which model features increased or reduced one order's risk.")
    demo_order_id = (
        st.query_params.get("order_id")
        if st.query_params.get("demo") == "explanation"
        else None
    )

    with st.form("explanation_form"):
        order_id = st.text_input(
            "Order ID",
            placeholder="be55f985440dddd650b389a55db8e49c",
            key="explanation_order_id",
            value=demo_order_id or "",
        ).strip()
        submitted = st.form_submit_button("Explain risk")

    if demo_order_id:
        submitted = True
    if not submitted:
        return
    if not order_id:
        st.warning("Please enter an order ID.")
        return

    payload, error, status_code = call_api(
        "GET", f"/orders/{order_id}/explanation"
    )
    if error:
        render_plain_error(error, status_code)
        return

    rows = []
    for item in payload["explanations"]:
        direction = (
            "Increases risk"
            if item["direction"] == "increases_risk"
            else "Reduces risk"
        )
        rows.append(
            {
                "Feature": item["feature"],
                "Direction": direction,
                "Magnitude": round(float(item["approximate_magnitude"]), 4),
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.markdown(f"<div class='summary-box'>{payload['summary']}</div>", unsafe_allow_html=True)
    st.warning("These are correlational signals, not causal explanations")


def _late_rate_color(rate: float | None) -> str:
    """Return traffic-light color for seller historical late rate."""
    if rate is None:
        return "#A0AEC0"
    if rate > 0.15:
        return "#E53E3E"
    if rate >= 0.08:
        return "#D69E2E"
    return "#38A169"


def render_seller_history_tab() -> None:
    """Render leakage-safe seller-history metrics."""
    st.subheader("Seller History")
    st.write("Review a seller's historical order volume, late rate, and reviews.")
    demo_seller_id = (
        st.query_params.get("seller_id") if st.query_params.get("demo") == "seller" else None
    )

    with st.form("seller_history_form"):
        seller_id = st.text_input(
            "Seller ID",
            value=demo_seller_id or "",
            placeholder="3078096983cf766a32a06257648502d1",
        ).strip()
        submitted = st.form_submit_button("Get seller history")

    if not submitted and not demo_seller_id:
        return
    if not seller_id:
        st.warning("Please enter a seller ID.")
        return

    payload, error, status_code = call_api("GET", f"/sellers/{seller_id}/history")
    if error:
        render_plain_error(error, status_code)
        return

    late_rate = payload["historical_late_rate"]
    late_rate_text = "Unavailable" if late_rate is None else f"{late_rate * 100:.2f}%"
    late_rate_color = _late_rate_color(late_rate)
    review_score = payload["historical_avg_review_score"]
    review_text = "Unavailable" if review_score is None else f"{review_score:.2f} / 5"

    col1, col2, col3 = st.columns(3)
    col1.metric("Historical order volume", f"{payload['historical_order_volume']:,}")
    col2.markdown(
        f"""
        <div class="mini-card">
          <div class="mini-label">Historical late rate</div>
          <div class="mini-value" style="color:{late_rate_color};">{late_rate_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col3.metric("Average review score", review_text)
    st.caption(payload["leakage_rule"])


def render_agent_tab() -> None:
    """Render natural-language agent queries backed by the FastAPI router."""
    st.subheader("Ask the Agent")
    st.warning("Agent queries call a paid LLM API. Each query costs a small amount.")

    example_single_tool = "Will order 6340164ffcc87a11dd0ad37d2551994c arrive late?"
    example_loop_back = (
        "First predict delay risk for order be55f985440dddd650b389a55db8e49c "
        "and if it is high, explain why."
    )
    example_clarify = "Will my order be late?"
    demo_query = (
        st.query_params.get("query")
        if st.query_params.get("demo") == "agent"
        else None
    )

    with st.form("agent_query_form"):
        query = st.text_area(
            "Natural-language question",
            placeholder=example_loop_back,
            value=demo_query or "",
            height=140,
        ).strip()
        st.caption("Examples:")
        st.code(example_single_tool, language=None)
        st.code(example_loop_back, language=None)
        st.code(example_clarify, language=None)
        submitted = st.form_submit_button("Ask agent")

    if demo_query:
        submitted = True
    if not submitted:
        return
    if not query:
        st.warning("Please enter a question.")
        return

    payload, error, status_code = call_api(
        "POST",
        "/agent/query",
        json={"query": query},
    )
    if error:
        render_plain_error(error, status_code)
        return

    st.markdown(
        f"<div class='answer-box'>{payload['answer']}</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Agent trace", expanded=False):
        tools_called = payload.get("tools_called", [])
        if tools_called:
            for index, tool_name in enumerate(tools_called, start=1):
                st.write(f"{index}. `{tool_name}`")
        else:
            st.write("No tools were called. The agent asked for clarification.")


def main() -> None:
    """Render the Phase 7 Streamlit demo shell."""
    st.set_page_config(
        page_title="Delivery Risk Intelligence",
        page_icon="🚚",
        layout="wide",
    )
    apply_design()
    render_sidebar()
    st.title("AI-Powered Delivery Risk Intelligence Agent")
    st.caption("Ask operational questions through a clean UI backed by FastAPI.")

    order_risk_tab, explanation_tab, seller_tab, agent_tab = st.tabs(
        ["Order Risk", "Risk Explanation", "Seller History", "Ask the Agent"]
    )
    with order_risk_tab:
        render_order_risk_tab()
    with explanation_tab:
        render_explanation_tab()
    with seller_tab:
        render_seller_history_tab()
    with agent_tab:
        render_agent_tab()
    render_footer()


if __name__ == "__main__":
    main()
