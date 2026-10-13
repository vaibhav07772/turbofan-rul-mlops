"""Tests for drift detection module."""

import numpy as np
import pandas as pd
import pytest

from app.monitoring.drift import load_drift_report, run_data_drift


@pytest.fixture()
def sample_dataframes():
    rng = np.random.default_rng(42)
    n = 100
    ref = pd.DataFrame({
        "sensor_1": rng.normal(500, 5, n),
        "sensor_2": rng.normal(600, 10, n),
        "RUL": rng.integers(0, 125, n),
    })
    # Current: similar distribution (no drift)
    cur = pd.DataFrame({
        "sensor_1": rng.normal(500, 5, n),
        "sensor_2": rng.normal(600, 10, n),
        "RUL": rng.integers(0, 125, n),
    })
    return ref, cur


def test_drift_runs_and_returns_summary(sample_dataframes, tmp_path):
    ref, cur = sample_dataframes
    output = tmp_path / "drift.json"

    summary = run_data_drift(ref, cur, target_col="RUL", output_path=output)

    assert "dataset_drift" in summary
    assert isinstance(summary["dataset_drift"], bool)
    assert summary["n_reference"] == 100
    assert summary["n_current"] == 100
    assert summary["n_features_analyzed"] >= 2
    assert output.exists()


def test_drift_detects_shifted_data(sample_dataframes, tmp_path):
    ref, _ = sample_dataframes
    # Heavy shift in current data → should detect drift
    shifted = pd.DataFrame({
        "sensor_1": np.random.default_rng(1).normal(800, 5, 100),
        "sensor_2": np.random.default_rng(2).normal(900, 10, 100),
        "RUL": np.random.default_rng(3).integers(0, 125, 100),
    })
    output = tmp_path / "drift_shifted.json"

    summary = run_data_drift(ref, shifted, target_col="RUL", output_path=output)
    assert summary["n_features_drifted"] > 0


def test_load_drift_report_missing(tmp_path):
    missing = tmp_path / "nope.json"
    result = load_drift_report(missing)
    assert result is None