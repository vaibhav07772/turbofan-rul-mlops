"""Generate a small synthetic sample mimicking NASA C-MAPSS format.

Real dataset: NASA C-MAPSS Turbofan Engine Degradation.
Download from: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

This script creates a small synthetic version so we can start building
the pipeline before downloading the real data.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# NASA C-MAPSS FD001 format:
# columns: unit, cycle, op_setting_1, op_setting_2, op_setting_3,
#          sensor_1 ... sensor_21
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
ALL_COLS = ["unit", "cycle"] + OP_COLS + SENSOR_COLS


def make_engine(unit_id: int, max_cycles: int) -> pd.DataFrame:
    """Create one engine's time-series with gradual sensor degradation."""
    cycles = np.arange(1, max_cycles + 1)
    health = 1.0 - (cycles / max_cycles)  # 1 → 0 degradation

    rows = {
        "unit": unit_id,
        "cycle": cycles,
        "op_setting_1": rng.normal(0.0, 0.001, max_cycles),
        "op_setting_2": rng.normal(0.0, 0.0005, max_cycles),
        "op_setting_3": np.full(max_cycles, 100.0),
    }

    # Sensors degrade over time with noise
    for i, col in enumerate(SENSOR_COLS):
        base = 500.0 + i * 10
        drift = (1.0 - health) * (20 + i)  # worsens with cycles
        noise = rng.normal(0, 0.5, max_cycles)
        rows[col] = base + drift + noise

    return pd.DataFrame(rows)


def main() -> None:
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)

    # 20 engines with different lifespans
    all_frames = []
    for unit_id in range(1, 21):
        max_cycles = int(rng.integers(120, 300))
        df = make_engine(unit_id, max_cycles)
        # Add true RUL for train set (NASA provides this in RUL_FD001.txt)
        df["RUL"] = (max_cycles - df["cycle"]).astype(int)
        all_frames.append(df)

    df = pd.concat(all_frames, ignore_index=True)

    train_file = out / "train_FD001.txt"
    df.to_csv(train_file, sep=" ", index=False, header=False)
    print(f"[sample] wrote {len(df)} rows from 20 engines → {train_file}")
    print(f"[sample] columns: unit, cycle, op_settings (3), sensors (21), RUL")


if __name__ == "__main__":
    main()