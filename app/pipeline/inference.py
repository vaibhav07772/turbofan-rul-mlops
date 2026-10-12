"""Inference pipeline: load model + feature columns, prepare input, predict RUL."""

from __future__ import annotations

import threading
from typing import Any

import joblib
import pandas as pd

from app.config import get_settings
from app.logging_config import get_logger
from app.pipeline.features import create_features
from app.pipeline.preprocessing import prepare_data

logger = get_logger("inference")

_MODEL_LOCK = threading.Lock()
_MODEL: Any = None
_FEATURE_COLUMNS: list[str] | None = None


def load_model(force_reload: bool = False) -> tuple[Any, list[str]]:
    """Load model + feature columns from disk. Cached."""
    global _MODEL, _FEATURE_COLUMNS

    with _MODEL_LOCK:
        if _MODEL is not None and _FEATURE_COLUMNS is not None and not force_reload:
            return _MODEL, _FEATURE_COLUMNS

        settings = get_settings()
        model_file = settings.model_dir / "model.pkl"
        feature_file = settings.model_dir / "feature_columns.pkl"

        if not model_file.exists():
            raise FileNotFoundError(
                f"Model not found: {model_file}. " f"Run: python scripts/train.py"
            )
        if not feature_file.exists():
            raise FileNotFoundError(
                f"Feature columns not found: {feature_file}. " f"Run: python scripts/train.py"
            )

        _MODEL = joblib.load(model_file)
        _FEATURE_COLUMNS = joblib.load(feature_file)

        logger.info(
            "model_loaded",
            model_file=str(model_file),
            n_features=len(_FEATURE_COLUMNS),
        )
        return _MODEL, _FEATURE_COLUMNS


def is_model_loaded() -> bool:
    return _MODEL is not None and _FEATURE_COLUMNS is not None


def predict_from_history(history_df: pd.DataFrame) -> dict:
    """
    Predict RUL from a single engine's sensor history.

    Args:
        history_df: DataFrame with columns: unit, cycle, op_setting_1..3,
                    sensor_1..21 (NASA format). Minimum ~10 cycles for features.

    Returns:
        Dict with predicted_rul, status, confidence, warnings.
    """
    model, feature_columns = load_model()

    if len(history_df) < 10:
        raise ValueError(f"Need at least 10 cycles for feature engineering, got {len(history_df)}")

    # Apply same preprocessing as training
    df = prepare_data(history_df, drop_flat=True)

    # Apply same feature engineering as training
    df = create_features(df, windows=(5, 10), lags=(1, 2))

    if len(df) == 0:
        raise ValueError(
            "Feature engineering produced 0 rows. " "Provide more cycles (recommended: 15+)."
        )

    # Check missing features
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features after engineering: {missing[:5]}...")

    # Take the last row (latest state)
    X = df[feature_columns].iloc[[-1]]

    # Predict
    predicted_rul = float(model.predict(X)[0])
    predicted_rul = max(0.0, predicted_rul)

    # Risk classification
    if predicted_rul < 20:
        status = "CRITICAL"
        message = "Engine is at risk of failure soon. Schedule maintenance immediately."
    elif predicted_rul < 50:
        status = "WARNING"
        message = "Engine shows degradation. Plan maintenance soon."
    else:
        status = "HEALTHY"
        message = "Engine is within acceptable operating range."

    return {
        "predicted_rul": round(predicted_rul, 2),
        "status": status,
        "message": message,
        "last_cycle": int(df["cycle"].iloc[-1]),
        "model_used": type(model).__name__,
        "n_features": len(feature_columns),
        "n_history_cycles": int(len(history_df)),
    }
