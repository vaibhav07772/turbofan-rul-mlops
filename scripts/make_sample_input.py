"""Generate a sample input CSV for testing the prediction API.

Creates a single engine's 30-cycle history in NASA format (no header).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]


def make_engine_history(unit_id: int = 99, n_cycles: int = 30) -> pd.DataFrame:
    """Make a synthetic engine with degradation (looks like mid-life)."""
    cycles = np.arange(1, n_cycles + 1)
    # Simulate degradation: engine at ~60% of its life
    health = 1.0 - (cycles / (n_cycles * 1.6))

    rows = {
        "unit": unit_id,
        "cycle": cycles,
        "op_setting_1": rng.normal(0.0, 0.001, n_cycles),
        "op_setting_2": rng.normal(0.0, 0.0005, n_cycles),
        "op_setting_3": np.full(n_cycles, 100.0),
    }

    for i, col in enumerate(SENSOR_COLS):
        base = 500.0 + i * 10
        drift = (1.0 - health) * (20 + i)
        noise = rng.normal(0, 0.5, n_cycles)
        rows[col] = base + drift + noise

    return pd.DataFrame(rows)


def main() -> None:
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "sample_input.txt"

    df = make_engine_history(unit_id=99, n_cycles=30)
    df.to_csv(out_file, sep=" ", index=False, header=False)

    print(f"[sample] wrote {len(df)} cycles from engine 99 → {out_file}")
    print(f"[sample] columns: {df.shape[1]} (unit, cycle, 3 op_settings, 21 sensors)")


if __name__ == "__main__":
    main()