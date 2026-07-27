"""Readiness endpoint and artifact validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.errors import AppError
from src.api.main import app
from src.api.readiness import verify_runtime_readiness


client = TestClient(app)


def test_ready_endpoint_loads_local_artifacts_without_llm() -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model": "logistic_regression",
        "feature_count": 23,
        "llm_checked": False,
    }


def test_missing_artifact_is_not_ready(tmp_path: Path) -> None:
    existing = tmp_path / "exists"
    existing.write_text("test", encoding="utf-8")
    with pytest.raises(AppError) as captured:
        verify_runtime_readiness(
            model_path=tmp_path / "missing-model",
            feature_contract_path=existing,
            feature_dataset_path=existing,
            model_loader=lambda: {},
            feature_loader=lambda: [],
        )
    assert captured.value.status_code == 503
    assert captured.value.category.value == "model_loading_error"


def test_loader_failure_is_safe_not_ready(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_text("test", encoding="utf-8")

    def fail_loader():
        raise ValueError("private artifact detail")

    with pytest.raises(AppError) as captured:
        verify_runtime_readiness(
            model_path=artifact,
            feature_contract_path=artifact,
            feature_dataset_path=artifact,
            model_loader=fail_loader,
            feature_loader=lambda: ["feature"],
        )
    assert captured.value.category.value == "model_loading_error"
    assert "private artifact detail" not in str(captured.value)
