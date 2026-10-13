"""Data drift detection using Evidently AI.

Compares the current validation/prediction distribution against
the reference (training) distribution.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger("drift")


def run_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    target_col: str | None = "RUL",
    feature_cols: list[str] | None = None,
    output_path: Path | None = None,
) -> dict:
    """
    Run Evidently data drift report between reference and current data.

    Args:
        reference_df: Training distribution (baseline).
        current_df: New/current data to compare.
        target_col: Target column name (optional).
        feature_cols: Subset of columns to analyze (optional).
        output_path: Where to save JSON report (optional).

    Returns:
        Dict with drift summary + per-feature results.
    """
    # Select only numeric features (Evidently doesn't handle all types well)
    if feature_cols is None:
        feature_cols = [
            c for c in reference_df.columns
            if c in current_df.columns
            and pd.api.types.is_numeric_dtype(reference_df[c])
            and c != target_col
        ]

    ref = reference_df[feature_cols].copy()
    cur = current_df[feature_cols].copy()

    # Column mapping
    column_mapping = ColumnMapping()
    if target_col and target_col in reference_df.columns and target_col in current_df.columns:
        ref[target_col] = reference_df[target_col]
        cur[target_col] = current_df[target_col]
        column_mapping.target = target_col

    # Build report
    metrics = [DataDriftPreset()]
    if target_col:
        metrics.append(TargetDriftPreset())

    report = Report(metrics=metrics)
    report.run(reference_data=ref, current_data=cur, column_mapping=column_mapping)

    result = report.as_dict()

    # Extract summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_reference": int(len(ref)),
        "n_current": int(len(cur)),
        "n_features_analyzed": len(feature_cols),
    }

    # Extract drift summary
    drifted_features = []
    dataset_drift = False

    for metric_result in result.get("metrics", []):
        metric_name = metric_result.get("metric", "")
        if metric_name == "DataDriftTable":
            metric_data = metric_result.get("result", {})
            dataset_drift = metric_data.get("dataset_drift", False)
            drift_by_feature = metric_data.get("drift_by_columns", {})

            for feature, info in drift_by_feature.items():
                if info.get("drift_detected", False):
                    drifted_features.append({
                        "feature": feature,
                        "drift_score": round(float(info.get("drift_score", 0.0)), 4),
                        "stattest": info.get("stattest_name", ""),
                        "threshold": info.get("threshold", 0.0),
                    })

    summary["dataset_drift"] = bool(dataset_drift)
    summary["n_features_drifted"] = len(drifted_features)
    summary["drift_share"] = round(
        len(drifted_features) / max(len(feature_cols), 1), 4
    )
    summary["drifted_features"] = sorted(
        drifted_features, key=lambda x: x["drift_score"], reverse=True
    )[:20]  # top 20

    # Save
    if output_path is None:
        settings = get_settings()
        output_path = settings.model_dir / "drift_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "full_report": result}, f, indent=2)

    logger.info(
        "drift_report_computed",
        dataset_drift=summary["dataset_drift"],
        drifted=summary["n_features_drifted"],
        total=summary["n_features_analyzed"],
    )

    return summary


def load_drift_report(path: Path | None = None) -> dict | None:
    """Load a saved drift report."""
    if path is None:
        settings = get_settings()
        path = settings.model_dir / "drift_report.json"
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data.get("summary")
    except Exception:
        return None