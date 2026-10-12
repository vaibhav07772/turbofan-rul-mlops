"""Load NASA C-MAPSS Turbofan datasets (space-separated, no header)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.pipeline.schema import BASE_COLS


def load_cmapss(
    path: str | Path,
    *,
    with_rul: bool = True,
    rul_cap: int | None = 125,
) -> pd.DataFrame:
    """
    Load a NASA C-MAPSS text file.

    Handles two formats:
    - 26 columns (standard NASA): unit, cycle, op_1..3, sensor_1..21
    - 27 columns (with RUL included): same + RUL column

    Args:
        path: Path to .txt file (space-separated, no header).
        with_rul: If True and RUL not in file, compute it from max cycle.
        rul_cap: Cap RUL at this value (piecewise-linear RUL). None = no cap.

    Returns:
        DataFrame with unit, cycle, op_settings, sensors, and optionally RUL.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

    n_cols = df.shape[1]
    base_n = len(BASE_COLS)  # 26

    if n_cols == base_n:
        df.columns = BASE_COLS
        has_rul_in_file = False
    elif n_cols == base_n + 1:
        df.columns = BASE_COLS + ["RUL"]
        has_rul_in_file = True
    else:
        raise ValueError(
            f"Unexpected column count: {n_cols}. "
            f"Expected {base_n} (NASA standard) or {base_n + 1} (with RUL)."
        )

    # Cast IDs to int
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    # Compute RUL if not already present
    if with_rul and not has_rul_in_file:
        max_cycle = df.groupby("unit")["cycle"].transform("max")
        df["RUL"] = (max_cycle - df["cycle"]).astype(int)

    # Optionally cap RUL (piecewise-linear target)
    if with_rul and rul_cap is not None and "RUL" in df.columns:
        df["RUL"] = df["RUL"].clip(upper=rul_cap)

    return df


def summarize(df: pd.DataFrame) -> dict:
    """Return a quick summary for logging."""
    info = {
        "rows": int(len(df)),
        "engines": int(df["unit"].nunique()),
        "min_cycle": int(df["cycle"].min()),
        "max_cycle": int(df["cycle"].max()),
        "columns": int(len(df.columns)),
    }
    if "RUL" in df.columns:
        info["rul_min"] = int(df["RUL"].min())
        info["rul_max"] = int(df["RUL"].max())
    return info