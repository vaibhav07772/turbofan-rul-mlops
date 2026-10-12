"""Train the RUL model and save it to disk.

Usage:
    python scripts/train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib

from app.config import get_settings
from app.logging_config import configure_logging, get_logger
from app.pipeline.data_loader import load_cmapss, summarize
from app.pipeline.features import create_features, get_feature_columns
from app.pipeline.models import MODEL_CANDIDATES
from app.pipeline.preprocessing import prepare_data, train_val_split_by_engine
from app.pipeline.schema import validate_schema
from app.pipeline.training import train_all_models

configure_logging()
logger = get_logger("train")


def main() -> None:
    settings = get_settings()
    print("=" * 55)
    print(" Turbofan RUL — Training")
    print("=" * 55)

    data_file = settings.raw_data_dir / "train_FD001.txt"
    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_file}\n"
            f"Run: python scripts/make_sample_data.py"
        )

    print(f"\n[1] Loading {data_file}")
    df = load_cmapss(data_file, with_rul=True, rul_cap=125)
    print(f"    {summarize(df)}")

    df = validate_schema(df, with_rul=True)
    print(f"[2] Validation: PASS ({len(df)} rows)")

    df = prepare_data(df, drop_flat=True)
    print(f"[3] Preprocessing: {df.shape}")

    df = create_features(df, windows=(5, 10), lags=(1, 2))
    print(f"[4] Feature engineering: {df.shape}")

    train_df, val_df = train_val_split_by_engine(df, val_ratio=0.2)
    feat_cols = get_feature_columns(train_df)
    print(f"[5] Split: train={train_df.shape}, val={val_df.shape}")
    print(f"    Feature columns: {len(feat_cols)}")

    X_train = train_df[feat_cols]
    y_train = train_df["RUL"].values
    X_val = val_df[feat_cols]
    y_val = val_df["RUL"].values

    print(f"\n[6] Training {len(MODEL_CANDIDATES)} models...")
    result = train_all_models(X_train, y_train, X_val, y_val)

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    model_file = settings.model_dir / "model.pkl"
    feature_file = settings.model_dir / "feature_columns.pkl"

    joblib.dump(result["best_model"], model_file)
    joblib.dump(feat_cols, feature_file)

    print(f"\n[7] Saved best model ({result['best_name']}) → {model_file}")
    print(f"    Feature columns → {feature_file}")
    print(f"    MLflow run_id: {result['mlflow_run_id']}")
    print(f"\n[✓] Training complete in {result['duration_s']}s")


if __name__ == "__main__":
    main()