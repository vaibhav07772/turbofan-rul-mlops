"""Compute data drift report between training and validation data.

Usage:
    python scripts/compute_drift.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.logging_config import configure_logging, get_logger
from app.monitoring.drift import run_data_drift
from app.pipeline.data_loader import load_cmapss
from app.pipeline.features import create_features, get_feature_columns
from app.pipeline.preprocessing import prepare_data, train_val_split_by_engine

configure_logging()
logger = get_logger("drift-script")


def main() -> None:
    settings = get_settings()
    print("=" * 55)
    print(" Data Drift Detection (Evidently)")
    print("=" * 55)

    data_file = settings.raw_data_dir / "train_FD001.txt"
    if not data_file.exists():
        print(f"[!] Dataset not found: {data_file}")
        return

    # Load + prepare (same pipeline)
    df = load_cmapss(data_file, with_rul=True, rul_cap=125)
    df = prepare_data(df, drop_flat=True)
    df = create_features(df, windows=(5, 10), lags=(1, 2))

    # Split
    train_df, val_df = train_val_split_by_engine(df, val_ratio=0.2)
    feat_cols = get_feature_columns(train_df)

    print(f"\n[1] Reference (train): {train_df.shape}")
    print(f"    Current (val):     {val_df.shape}")
    print(f"    Features:          {len(feat_cols)}")

    print("\n[2] Running drift detection (may take 30-60 sec)...")
    summary = run_data_drift(
        reference_df=train_df,
        current_df=val_df,
        target_col="RUL",
        feature_cols=feat_cols,
    )

    print(f"\n[3] Results:")
    print(f"    Dataset drift detected: {summary['dataset_drift']}")
    print(f"    Features drifted: {summary['n_features_drifted']} / {summary['n_features_analyzed']}")
    print(f"    Drift share: {summary['drift_share']:.2%}")

    if summary["drifted_features"]:
        print(f"\n[4] Top 5 drifted features:")
        for item in summary["drifted_features"][:5]:
            print(f"    - {item['feature']:40s} score={item['drift_score']:.4f}")

    output = settings.model_dir / "drift_report.json"
    print(f"\n[5] Saved → {output}")
    print(f"\n[✓] Drift detection complete")


if __name__ == "__main__":
    main()