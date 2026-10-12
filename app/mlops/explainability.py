"""SHAP-based model explainability for Turbofan RUL predictions."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger("explainability")


def compute_shap_importance(
    model,
    X_sample: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 20,
    max_samples: int = 100,
) -> dict:
    """
    Compute global SHAP feature importances using TreeExplainer.

    For tree-based models (RF, XGB, GB) this is fast.
    Uses a sample of rows to keep it quick.
    """
    if len(X_sample) > max_samples:
        X_sample = X_sample.sample(max_samples, random_state=42)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        # Fallback: use generic explainer with limited samples
        logger.warning("tree_explainer_failed_using_kernel")
        explainer = shap.Explainer(model.predict, X_sample.head(50))
        shap_values = explainer(X_sample.head(50)).values

    # For regression, shap_values is 2D array
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Global importance = mean(|SHAP|) per feature
    mean_abs = np.abs(shap_values).mean(axis=0)

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    top = importance.head(top_n)

    return {
        "top_features": [
            {"feature": row["feature"], "importance": round(float(row["importance"]), 4)}
            for _, row in top.iterrows()
        ],
        "n_features": len(feature_names),
        "n_samples_explained": int(len(X_sample)),
    }


def compute_local_explanation(
    model,
    X_single: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 10,
) -> dict:
    """Explain a single prediction — which features pushed the RUL up/down."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_single)
    except Exception:
        explainer = shap.Explainer(model.predict, X_single)
        shap_values = explainer(X_single).values

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    values = shap_values[0]

    # Sort by absolute contribution
    order = np.argsort(np.abs(values))[::-1][:top_n]

    return {
        "predicted_rul": float(model.predict(X_single)[0]),
        "top_contributions": [
            {
                "feature": feature_names[i],
                "value": round(float(X_single.iloc[0, i]), 4),
                "contribution": round(float(values[i]), 4),
            }
            for i in order
        ],
    }


def save_shap_report(report: dict, output_path: Path | None = None) -> Path:
    """Save the SHAP importance report as JSON."""
    if output_path is None:
        settings = get_settings()
        output_path = settings.model_dir / "shap_importance.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return output_path


def load_shap_report(path: Path | None = None) -> dict | None:
    """Load a saved SHAP importance report."""
    if path is None:
        settings = get_settings()
        path = settings.model_dir / "shap_importance.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)