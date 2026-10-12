"""Generate a small synthetic sample mimicking NASA C-MAPSS format.

Real dataset: NASA C-MAPSS Turbofan Engine Degradation.
Download from: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

NASA format: space-separated, no header, no RUL column.
RUL is computed later from max cycle per engine.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]


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

    for i, col in enumerate(SENSOR_COLS):
        base = 500.0 + i * 10
        drift = (1.0 - health) * (20 + i)
        noise = rng.normal(0, 0.5, max_cycles)
        rows[col] = base + drift + noise

    return pd.DataFrame(rows)


def main() -> None:
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)

    all_frames = []
    for unit_id in range(1, 21):
        max_cycles = int(rng.integers(120, 300))
        all_frames.append(make_engine(unit_id, max_cycles))

    df = pd.concat(all_frames, ignore_index=True)

    # NOTE: No RUL column — matches NASA format exactly.
    # Expected columns: unit, cycle, op_1..3, sensor_1..21 → 26 columns
    out_file = out / "train_FD001.txt"
    df.to_csv(out_file, sep=" ", index=False, header=False)

    print(f"[sample] wrote {len(df)} rows from 20 engines → {out_file}")
    print(f"[sample] columns: {df.shape[1]} (unit, cycle, 3 op_settings, 21 sensors)")


if __name__ == "__main__":
    main()