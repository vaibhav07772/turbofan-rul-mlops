"""Training loop: evaluate 3 models, pick best by RMSE, log to MLflow."""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.mlops.mlflow_utils import is_enabled, log_training_run
from app.pipeline.models import MODEL_CANDIDATES, build_model


def evaluate(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "R2": round(r2, 4),
    }


def train_all_models(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    dataset_key: str = "turbofan-fd001",
):
    """Train all candidate models, evaluate, select best, log to MLflow."""
    started = time.perf_counter()
    results: dict[str, dict] = {}
    fitted_models: dict[str, object] = {}

    for model_name, candidates in MODEL_CANDIDATES.items():
        params = candidates[0]
        model = build_model(model_name, params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        metrics = evaluate(y_val, pred)
        metrics["params"] = params
        results[model_name] = metrics
        fitted_models[model_name] = model
        print(
            f"[TRAIN] {model_name:20s} "
            f"RMSE={metrics['RMSE']:.3f}  "
            f"MAE={metrics['MAE']:.3f}  "
            f"R2={metrics['R2']:.4f}"
        )

    best_name = min(results, key=lambda k: results[k]["RMSE"])
    best_model = fitted_models[best_name]
    best_params = MODEL_CANDIDATES[best_name][0]

    elapsed = time.perf_counter() - started
    print(
        f"\n[TRAIN] best={best_name} "
        f"RMSE={results[best_name]['RMSE']} | "
        f"total={elapsed:.1f}s"
    )

    mlflow_run_id = None
    if is_enabled():
        try:
            mlflow_run_id = log_training_run(
                dataset_key=dataset_key,
                n_train=int(len(X_train)),
                n_val=int(len(X_val)),
                n_features=int(X_train.shape[1]),
                results=results,
                best_name=best_name,
                best_params=best_params,
                best_model=best_model,
                X_train=X_train,
                y_train=np.asarray(y_train),
            )
        except Exception as exc:
            print(f"[MLFLOW] logging failed (non-fatal): {type(exc).__name__}: {exc}")

    return {
        "best_model": best_model,
        "best_name": best_name,
        "best_params": best_params,
        "results": results,
        "mlflow_run_id": mlflow_run_id,
        "duration_s": round(elapsed, 2),
    }
