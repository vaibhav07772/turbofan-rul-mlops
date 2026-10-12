"""Quick smoke test for the data pipeline."""

import sys
from pathlib import Path

# Ensure project root is importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.data_loader import load_cmapss, summarize
from app.pipeline.preprocessing import prepare_data, train_val_split_by_engine
from app.pipeline.features import create_features, get_feature_columns
from app.pipeline.schema import validate_schema

print("=" * 50)
print(" Pipeline Smoke Test")
print("=" * 50)

# 1. Load
df = load_cmapss("data/raw/train_FD001.txt", with_rul=True, rul_cap=125)
print(f"\n[1] Loaded: {summarize(df)}")

# 2. Validate
df = validate_schema(df, with_rul=True)
print(f"[2] Validation: PASS ({len(df)} rows)")

# 3. Preprocess
df = prepare_data(df, drop_flat=True)
print(f"[3] After preprocessing: {df.shape}")

# 4. Features
df = create_features(df, windows=(5, 10), lags=(1, 2))
print(f"[4] After features: {df.shape}")

# 5. Get feature cols
feat_cols = get_feature_columns(df)
print(f"[5] Feature columns: {len(feat_cols)}")

# 6. Split
train, val = train_val_split_by_engine(df, val_ratio=0.2)
print(f"[6] Train: {train.shape} | Val: {val.shape}")
print(f"     Train engines: {train['unit'].nunique()}")
print(f"     Val engines:   {val['unit'].nunique()}")

print("\n[✓] Pipeline works end-to-end!")