"""Preprocessing: drop useless sensors, normalize, split train/val."""

from __future__ import annotations

import pandas as pd

from app.pipeline.schema import OP_SETTING_COLS, SENSOR_COLS


# Sensors that are flat (constant) in FD001 — drop them.
# Source: NASA C-MAPSS documentation + community analysis.
CONSTANT_SENSORS_FD001 = {
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
}


def drop_constant_sensors(
    df: pd.DataFrame, to_drop: set[str] | None = None
) -> pd.DataFrame:
    """Drop sensor columns that carry no signal (flat in FD001)."""
    if to_drop is None:
        to_drop = CONSTANT_SENSORS_FD001
    drop_cols = [c for c in to_drop if c in df.columns]
    return df.drop(columns=drop_cols, errors="ignore")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns (excludes unit, cycle, RUL)."""
    excluded = {"unit", "cycle", "RUL"}
    return [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def train_val_split_by_engine(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by engine unit (NOT random rows) so engines in val are unseen.
    This is critical for RUL — random row splitting leaks future info.
    """
    units = df["unit"].unique()
    n_val = max(1, int(len(units) * val_ratio))

    rng = pd.Series(units).sample(frac=1.0, random_state=seed).to_numpy()
    val_units = set(rng[:n_val])
    train_units = set(rng[n_val:])

    train_df = df[df["unit"].isin(train_units)].reset_index(drop=True)
    val_df = df[df["unit"].isin(val_units)].reset_index(drop=True)
    return train_df, val_df


def prepare_data(df: pd.DataFrame, drop_flat: bool = True) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    data = df.copy()
    if drop_flat:
        data = drop_constant_sensors(data)
    data = data.sort_values(["unit", "cycle"]).reset_index(drop=True)
    return data