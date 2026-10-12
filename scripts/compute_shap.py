"""Compute SHAP feature importances for the trained model.

Usage:
    python scripts/compute_shap.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib

from app.config import get_settings
from app.logging_config import configure_logging, get_logger
from app.mlops.explainability import compute_shap_importance, save_shap_report
from app.pipeline.data_loader import load_cmapss
from app.pipeline.features import create_features, get_feature_columns
from app.pipeline.preprocessing import prepare_data, train_val_split_by_engine

configure_logging()
logger = get_logger("shap")


def main() -> None:
    settings = get_settings()
    print("=" * 55)
    print(" SHAP Feature Importance")
    print("=" * 55)

    # Load model
    model_file = settings.model_dir / "model.pkl"
    feature_file = settings.model_dir / "feature_columns.pkl"

    if not model_file.exists() or not feature_file.exists():
        print(f"[!] Model not found: {model_file}")
        print("[!] Run: python scripts/train.py")
        return

    model = joblib.load(model_file)
    feature_cols = joblib.load(feature_file)
    print(f"\n[1] Model loaded: {type(model).__name__}")
    print(f"    Features: {len(feature_cols)}")

    # Load + prepare data (same pipeline as training)
    data_file = settings.raw_data_dir / "train_FD001.txt"
    df = load_cmapss(data_file, with_rul=True, rul_cap=125)
    df = prepare_data(df, drop_flat=True)
    df = create_features(df, windows=(5, 10), lags=(1, 2))

    train_df, val_df = train_val_split_by_engine(df, val_ratio=0.2)
    X_val = val_df[feature_cols]
    print(f"[2] Validation data: {X_val.shape}")

    # Compute SHAP
    print("[3] Computing SHAP values (this may take 1-2 min)...")
    report = compute_shap_importance(
        model=model,
        X_sample=X_val,
        feature_names=feature_cols,
        top_n=20,
        max_samples=100,
    )

    print(f"\n[4] Top 10 features:")
    print("-" * 55)
    for i, item in enumerate(report["top_features"][:10], 1):
        print(f"  {i:2d}. {item['feature']:40s} {item['importance']:.4f}")
    print("-" * 55)

    # Save
    output = save_shap_report(report)
    print(f"\n[5] Saved SHAP report → {output}")
    print(f"\n[✓] SHAP importance computed ({report['n_samples_explained']} samples)")


if __name__ == "__main__":
    main()