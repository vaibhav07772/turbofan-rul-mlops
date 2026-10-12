"""Shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.pipeline.schema import BASE_COLS, SENSOR_COLS


@pytest.fixture(scope="session")
def sample_engine_df() -> pd.DataFrame:
    """A single engine's 30-cycle history (NASA format)."""
    rng = np.random.default_rng(42)
    n = 30

    rows = {
        "unit": [99] * n,
        "cycle": list(range(1, n + 1)),
        "op_setting_1": rng.normal(0.0, 0.001, n),
        "op_setting_2": rng.normal(0.0, 0.0005, n),
        "op_setting_3": [100.0] * n,
    }
    for i, col in enumerate(SENSOR_COLS):
        base = 500.0 + i * 10
        drift = np.linspace(0, 20 + i, n)
        rows[col] = base + drift + rng.normal(0, 0.5, n)

    return pd.DataFrame(rows)[BASE_COLS]


@pytest.fixture(scope="session")
def sample_engine_file(tmp_path_factory, sample_engine_df) -> str:
    """Save the sample engine to a temp file (space-separated, no header)."""
    path = tmp_path_factory.mktemp("data") / "engine.txt"
    sample_engine_df.to_csv(path, sep=" ", index=False, header=False)
    return str(path)


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    """FastAPI TestClient with MLflow disabled."""
    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    from app.config import get_settings

    get_settings.cache_clear()

    from app.main import app

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()
