"""Time-series feature engineering for RUL prediction.

Optimized version: builds all feature columns in a dict and concatenates
once, avoiding pandas' "highly fragmented DataFrame" warning.
"""

from __future__ import annotations

import pandas as pd

from app.pipeline.schema import OP_SETTING_COLS, SENSOR_COLS


def create_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10),
    lags: tuple[int, ...] = (1, 2),
) -> pd.DataFrame:
    """
    Create rolling/lag features per engine.

    IMPORTANT: groupby('unit') so features don't cross engine boundaries.
    """
    data = df.copy()
    sensor_cols = [c for c in SENSOR_COLS if c in data.columns]

    g = data.groupby("unit", sort=False)

    # Build all new columns in a dict (avoids DataFrame fragmentation)
    new_cols: dict[str, pd.Series] = {}

    # Rolling mean / std
    for window in windows:
        for col in sensor_cols:
            rolled = g[col].rolling(window)
            new_cols[f"{col}_roll_mean_{window}"] = rolled.mean().reset_index(
                level=0, drop=True
            )
            new_cols[f"{col}_roll_std_{window}"] = rolled.std().reset_index(
                level=0, drop=True
            )

    # Lags
    for lag in lags:
        for col in sensor_cols:
            new_cols[f"{col}_lag_{lag}"] = g[col].shift(lag)

    # Rate of change
    for col in sensor_cols:
        new_cols[f"{col}_diff"] = g[col].diff()

    # Concatenate once — much faster
    features_df = pd.DataFrame(new_cols, index=data.index)
    data = pd.concat([data, features_df], axis=1)

    # Drop NaNs created by rolling/lag
    data = data.dropna().reset_index(drop=True)
    return data


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all feature columns (excludes unit, cycle, RUL)."""
    excluded = {"unit", "cycle", "RUL"}
    return [c for c in df.columns if c not in excluded]