"""Tests for the data pipeline."""

import numpy as np
import pandas as pd

from app.pipeline.features import create_features, get_feature_columns
from app.pipeline.preprocessing import (
    drop_constant_sensors,
    prepare_data,
    train_val_split_by_engine,
)
from app.pipeline.schema import BASE_COLS, SENSOR_COLS, validate_schema


def test_validate_schema_passes(sample_engine_df):
    df = validate_schema(sample_engine_df.assign(RUL=50), with_rul=True)
    assert "RUL" in df.columns


def test_validate_schema_rejects_negative_rul(sample_engine_df):
    import pytest

    from app.pipeline.schema import SchemaValidationError

    bad = sample_engine_df.assign(RUL=-5)
    with pytest.raises(SchemaValidationError):
        validate_schema(bad, with_rul=True)


def test_validate_schema_rejects_missing_cols():
    import pytest

    from app.pipeline.schema import SchemaValidationError

    bad = pd.DataFrame({"unit": [1], "cycle": [1]})
    with pytest.raises(SchemaValidationError):
        validate_schema(bad, with_rul=False)


def test_drop_constant_sensors(sample_engine_df):
    before = sample_engine_df.shape[1]
    after = drop_constant_sensors(sample_engine_df).shape[1]
    assert after < before


def test_prepare_data_sorts_by_unit_cycle(sample_engine_df):
    df = sample_engine_df.sample(frac=1.0, random_state=1)
    out = prepare_data(df, drop_flat=False)
    assert out["cycle"].is_monotonic_increasing


def test_create_features_adds_columns(sample_engine_df):
    df = sample_engine_df.copy()
    df["RUL"] = 100
    feat = create_features(df, windows=(5,), lags=(1,))
    assert feat.shape[1] > df.shape[1]


def test_feature_columns_excludes_metadata(sample_engine_df):
    df = sample_engine_df.assign(RUL=50)
    df = create_features(df, windows=(5,), lags=(1,))
    cols = get_feature_columns(df)
    assert "unit" not in cols
    assert "cycle" not in cols
    assert "RUL" not in cols


def test_train_val_split_by_engine(sample_engine_df):
    # Create 5 engines by concatenating with different unit IDs
    frames = []
    for u in range(1, 6):
        f = sample_engine_df.copy()
        f["unit"] = u
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["RUL"] = 50

    train, val = train_val_split_by_engine(df, val_ratio=0.2)
    assert train["unit"].nunique() == 4
    assert val["unit"].nunique() == 1
    assert set(train["unit"]).isdisjoint(set(val["unit"]))


def test_schema_columns_match_nasa():
    assert len(BASE_COLS) == 26
    assert len(SENSOR_COLS) == 21
