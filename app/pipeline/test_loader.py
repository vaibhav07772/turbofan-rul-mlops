"""Loader for NASA C-MAPSS test data + RUL ground truth."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.pipeline.schema import BASE_COLS


def load_test_data(
    test_path: str | Path,
    rul_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load NASA C-MAPSS test file (with true RUL joined if available).

    Test format: same as train but NO RUL column.
    RUL_FD001.txt: 1 column, one RUL value per engine.
    """
    test_path = Path(test_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    df = pd.read_csv(test_path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != len(BASE_COLS):
        raise ValueError(f"Expected {len(BASE_COLS)} columns, got {df.shape[1]}")
    df.columns = BASE_COLS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    if rul_path is None:
        return df

    rul_path = Path(rul_path)
    if not rul_path.exists():
        raise FileNotFoundError(f"RUL file not found: {rul_path}")

    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, engine="python")
    rul_df.columns = ["RUL"]

    # Assign RUL to the LAST cycle of each engine only
    last_cycles = df.groupby("unit")["cycle"].max().reset_index()
    last_cycles["true_RUL"] = rul_df["RUL"].values
    df = df.merge(last_cycles[["unit", "true_RUL"]], on="unit", how="left")
    return df


def get_last_cycle_per_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per engine (last cycle)."""
    idx = df.groupby("unit")["cycle"].idxmax()
    return df.loc[idx].reset_index(drop=True)