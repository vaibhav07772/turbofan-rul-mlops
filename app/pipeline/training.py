"""Training loop: evaluate 3 regression models + 3-class classification metrics."""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from app.mlops.mlflow_utils import is_enabled, log_training_run
from app.pipeline.classification import CLASSES, ruls_to_classes
from app.pipeline.models import MODEL_CANDIDATES, build_model


def evaluate_regression(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "R2": round(r2, 4),
    }


def evaluate_classification(y_true_rul, y_pred_rul) -> dict:
    """Treat RUL predictions as 3-class classification problem."""
    y_true_cls = ruls_to_classes(y_true_rul)
    y_pred_cls = ruls_to_classes(y_pred_rul)

    # Per-class metrics (macro average)
    precision = float(precision_score(y_true_cls, y_pred_cls, average="macro", zero_division=0))
    recall = float(recall_score(y_true_cls, y_pred_cls, average="macro", zero_division=0))
    f1 = float(f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_true_cls, y_pred_cls))

    # Confusion matrix
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=CLASSES).tolist()

    # Per-class breakdown
    report = classification_report(
        y_true_cls, y_pred_cls, labels=CLASSES, output_dict=True, zero_division=0
    )

    return {
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1": round(f1 * 100, 2),
        "ConfusionMatrix": cm,
        "ClassReport": report,
        "Classes": CLASSES,
    }


def train_all_models(X_train, y_train, X_val, y_val, *, dataset_key: str = "turbofan-fd001"):
    """Train all candidate models, evaluate regression + classification, pick best."""
    started = time.perf_counter()
    results: dict[str, dict] = {}
    fitted_models: dict[str, object] = {}

    print("\n[REGRESSION + CLASSIFICATION METRICS]")
    print("-" * 90)
    for model_name, candidates in MODEL_CANDIDATES.items():
        params = candidates[0]
        model = build_model(model_name, params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        reg_metrics = evaluate_regression(y_val, pred)
        cls_metrics = evaluate_classification(y_val, pred)

        results[model_name] = {
            **reg_metrics,
            "Accuracy": cls_metrics["Accuracy"],
            "Precision": cls_metrics["Precision"],
            "Recall": cls_metrics["Recall"],
            "F1": cls_metrics["F1"],
            "params": params,
        }
        fitted_models[model_name] = model

        print(
            f"  {model_name:20s} "
            f"RMSE={reg_metrics['RMSE']:>7.3f}  "
            f"R2={reg_metrics['R2']:>7.4f}  "
            f"Acc={cls_metrics['Accuracy']:>6.2f}%  "
            f"F1={cls_metrics['F1']:>6.2f}%"
        )
    print("-" * 90)

    best_name = min(results, key=lambda k: results[k]["RMSE"])
    best_model = fitted_models[best_name]
    best_params = MODEL_CANDIDATES[best_name][0]

    elapsed = time.perf_counter() - started
    print(
        f"\n[TRAIN] best={best_name} "
        f"RMSE={results[best_name]['RMSE']} | "
        f"Accuracy={results[best_name]['Accuracy']}% | "
        f"F1={results[best_name]['F1']}% | "
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