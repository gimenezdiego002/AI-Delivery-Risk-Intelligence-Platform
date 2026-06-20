"""Deterministic tests for the Phase 4 delivery-risk tools."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.agent.tools import (
    explain_risk,
    get_seller_history,
    get_similar_past_orders,
    predict_delay_risk,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "delivery_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_delivery_risk_model.joblib"
FEATURES_PATH = PROJECT_ROOT / "models" / "model_features.json"


class AgentToolTests(unittest.TestCase):
    """Test tool contracts, leakage boundaries, and deterministic outputs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.features = json.loads(FEATURES_PATH.read_text())["features"]
        cls.data = pd.read_csv(
            DATA_PATH,
            parse_dates=[
                "order_purchase_timestamp",
                "order_delivered_customer_date",
            ],
        )
        cls.model = joblib.load(MODEL_PATH)["model"]

        eligible = cls.data.loc[
            cls.data["order_purchase_timestamp"] >= "2018-05-01"
        ].dropna(subset=cls.features)
        probabilities = cls.model.predict_proba(eligible[cls.features])[:, 1]
        strictly_probabilistic = (probabilities > 0) & (probabilities < 1)
        cls.valid_order = eligible.loc[strictly_probabilistic].iloc[0]

    def test_valid_order_returns_strict_probability(self) -> None:
        """A complete real order should return a non-degenerate probability."""
        result = predict_delay_risk(self.valid_order["order_id"])
        self.assertTrue(result["ok"])
        self.assertGreater(result["late_delivery_probability"], 0)
        self.assertLess(result["late_delivery_probability"], 1)
        self.assertIn(result["risk_level"], {"high", "low"})

    def test_unknown_order_returns_structured_error(self) -> None:
        """Unknown IDs should not produce unhandled exceptions."""
        result = predict_delay_risk("definitely-not-a-real-olist-order")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "order_not_found")
        self.assertIn("message", result["error"])

    def test_seller_history_excludes_later_orders(self) -> None:
        """A mid-history snapshot must exclude every later seller order."""
        seller_id = self.data["seller_id"].value_counts().index[0]
        seller_orders = self.data.loc[
            self.data["seller_id"] == seller_id
        ].sort_values("order_purchase_timestamp")
        snapshot = seller_orders.iloc[len(seller_orders) // 2]

        prior_orders = seller_orders.loc[
            seller_orders["order_purchase_timestamp"]
            < snapshot["order_purchase_timestamp"]
        ]
        later_orders = seller_orders.loc[
            seller_orders["order_purchase_timestamp"]
            > snapshot["order_purchase_timestamp"]
        ]
        self.assertGreater(len(later_orders), 0)

        result = get_seller_history(seller_id, snapshot["order_id"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["historical_order_volume"], len(prior_orders))
        self.assertLess(
            result["historical_order_volume"], len(prior_orders) + len(later_orders)
        )
        self.assertEqual(
            result["historical_late_rate"],
            snapshot["seller_historical_late_rate"],
        )

    def test_similar_orders_exclude_query_and_future_outcomes(self) -> None:
        """Neighbors must be distinct and completed before query time."""
        query_id = self.valid_order["order_id"]
        query_time = self.valid_order["order_purchase_timestamp"]
        result = get_similar_past_orders(query_id, top_n=3)
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["similar_orders"]), 3)
        for neighbor in result["similar_orders"]:
            self.assertNotEqual(neighbor["order_id"], query_id)
            self.assertLess(
                pd.Timestamp(neighbor["order_delivered_customer_date"]),
                query_time,
            )
            self.assertIsInstance(neighbor["was_late"], bool)

    def test_explanation_uses_only_contract_features(self) -> None:
        """Every explanation name must come from the persisted contract."""
        result = explain_risk(self.valid_order["order_id"])
        self.assertTrue(result["ok"])
        explained_features = {
            item["feature"] for item in result["explanations"]
        }
        self.assertTrue(explained_features)
        self.assertTrue(explained_features.issubset(set(self.features)))
        self.assertIn("not causal", result["caveat"])


if __name__ == "__main__":
    unittest.main()
