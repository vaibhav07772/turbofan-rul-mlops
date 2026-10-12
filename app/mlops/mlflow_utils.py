"""MLflow tracking + model registry helpers for Turbofan RUL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger("mlflow")


def is_enabled() -> bool:
    return get_settings().mlflow_enabled


def setup_mlflow() -> None:
    """Configure MLflow tracking URI + experiment. Safe to call repeatedly."""
    if not is_enabled():
        logger.info("mlflow_disabled")
        return

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name(settings.mlflow_experiment_name) is None:
        artifact_path = Path(settings.mlflow_artifact_location).resolve()
        artifact_path.mkdir(parents=True, exist_ok=True)
        client.create_experiment(
            settings.mlflow_experiment_name,
            artifact_location=artifact_path.as_uri(),
        )

    mlflow.set_experiment(settings.mlflow_experiment_name)
    logger.info(
        "mlflow_configured",
        tracking_uri=settings.mlflow_tracking_uri,
        experiment=settings.mlflow_experiment_name,
    )


def log_training_run(
    *,
    dataset_key: str,
    n_train: int,
    n_val: int,
    n_features: int,
    results: dict[str, dict[str, Any]],
    best_name: str,
    best_params: dict[str, Any],
    best_model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    extra_tags: dict[str, str] | None = None,
) -> str | None:
    """Log a full training session to MLflow."""
    if not is_enabled():
        return None

    setup_mlflow()
    settings = get_settings()

    parent_tags = {
        "dataset_key": dataset_key,
        "best_model": best_name,
        "environment": settings.environment,
        "app_version": settings.app_version,
    }
    if extra_tags:
        parent_tags.update(extra_tags)

    with mlflow.start_run(run_name=f"training-{best_name}") as parent:
        mlflow.set_tags(parent_tags)
        mlflow.log_params({
            "n_train": n_train,
            "n_val": n_val,
            "n_features": n_features,
            "candidate_models": ",".join(results.keys()),
            "best_model": best_name,
        })

        for model_name, metrics in results.items():
            with mlflow.start_run(run_name=model_name, nested=True):
                mlflow.set_tag("model_family", model_name)
                mlflow.log_param("params", metrics.get("params", {}))
                mlflow.log_metrics({
                    "rmse": float(metrics.get("RMSE", 0.0)),
                    "mae": float(metrics.get("MAE", 0.0)),
                    "r2": float(metrics.get("R2", 0.0)),
                })

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_rmse", float(results[best_name].get("RMSE", 0.0)))
        mlflow.log_metric("best_mae", float(results[best_name].get("MAE", 0.0)))
        mlflow.log_metric("best_r2", float(results[best_name].get("R2", 0.0)))

        try:
            signature = infer_signature(X_train, best_model.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(3),
                registered_model_name=(
                    settings.mlflow_registered_model_name
                    if settings.mlflow_auto_register
                    else None
                ),
            )
        except Exception:
            logger.exception("mlflow_log_model_failed", model=best_name)

        logger.info(
            "mlflow_run_logged",
            run_id=parent.info.run_id,
            best_model=best_name,
            registered=settings.mlflow_auto_register,
        )
        return parent.info.run_id