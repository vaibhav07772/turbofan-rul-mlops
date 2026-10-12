"""FastAPI app for Turbofan RUL prediction."""

import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.logging_config import configure_logging, get_logger
from app.middleware import RequestContextMiddleware
from app.monitoring import metrics
from app.monitoring.predictions import log_prediction
from app.pipeline.inference import is_model_loaded, load_model, predict_from_history

# --- Setup ---
configure_logging()
logger = get_logger("turbofan-rul")
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Predictive Maintenance: RUL Prediction for Turbofan Engines",
)

app.add_middleware(RequestContextMiddleware)


# --- Request models ---
class PredictRequest(BaseModel):
    """Predict from a single cycle's sensor values (needs ~10 cycles history)."""

    unit: int = Field(..., ge=1)
    cycle: int = Field(..., ge=1)
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensors: list[float] = Field(..., min_length=21, max_length=21)


# --- Startup ---
@app.on_event("startup")
async def startup_load_model():
    """Try to load the model at startup (non-fatal if missing)."""
    try:
        load_model()
        logger.info("model_loaded_at_startup")
    except FileNotFoundError as exc:
        logger.warning("model_not_found_at_startup", error=str(exc))


# --- Root ---
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
      <head><title>Turbofan RUL MLOps</title></head>
      <body style="font-family: sans-serif; padding: 2rem;">
        <h1>🚀 Turbofan RUL MLOps</h1>
        <p>Predictive Maintenance for Jet Engines</p>
        <ul>
          <li><a href="/docs">API Docs (Swagger)</a></li>
          <li><a href="/health">Health</a></li>
          <li><a href="/metrics">Prometheus Metrics</a></li>
        </ul>
      </body>
    </html>
    """


# --- Prometheus ---
@app.get("/metrics")
async def prometheus_metrics():
    payload, content_type = metrics.render_metrics()
    return Response(content=payload, media_type=content_type)


# --- Health ---
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "model_loaded": is_model_loaded(),
    }


@app.get("/health/live")
async def health_live():
    return {"status": "alive"}


@app.get("/health/ready")
async def health_ready():
    checks = {
        "data_dir": settings.data_dir.exists(),
        "model_dir": settings.model_dir.exists(),
        "model_loaded": is_model_loaded(),
    }
    ready = all(checks.values())
    return {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
    }


# --- Prediction: CSV upload (recommended) ---
@app.post("/api/predict/upload")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Upload a CSV with one engine's history. Returns predicted RUL.

    Expected format (space-separated, no header — NASA C-MAPSS style):
        unit cycle op_setting_1 op_setting_2 op_setting_3 sensor_1 ... sensor_21

    Also supports comma-separated with the same column order.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please select a file.")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".csv", ".txt"}:
        raise HTTPException(status_code=400, detail="Only .csv or .txt allowed.")

    started = time.perf_counter()

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file.")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read file.")

    try:
        # Try whitespace first (NASA format), then comma
        from io import BytesIO

        buffer = BytesIO(content)
        try:
            df = pd.read_csv(buffer, sep=r"\s+", header=None, engine="python")
            if df.shape[1] != 26:
                raise ValueError("Not space-separated")
        except Exception:
            buffer.seek(0)
            df = pd.read_csv(buffer)

        # If header was included, drop it
        if df.iloc[0, 0] == "unit" or str(df.iloc[0, 0]).lower().startswith("unit"):
            df = df.iloc[1:].reset_index(drop=True)

        from app.pipeline.schema import BASE_COLS

        if df.shape[1] != len(BASE_COLS):
            raise ValueError(
                f"Expected {len(BASE_COLS)} columns, got {df.shape[1]}. "
                f"Columns: unit, cycle, op_1..3, sensor_1..21"
            )

        df.columns = BASE_COLS
        df["unit"] = df["unit"].astype(int)
        df["cycle"] = df["cycle"].astype(int)
        for col in df.columns:
            if col not in {"unit", "cycle"}:
                df[col] = df[col].astype(float)

        if df["unit"].nunique() > 1:
            raise ValueError("Upload only one engine's history at a time.")

        result = await run_in_threadpool(predict_from_history, df)

        elapsed = time.perf_counter() - started

        # Metrics
        try:
            metrics.predictions_total.labels(status="success").inc()
            metrics.prediction_duration_seconds.observe(elapsed)
            metrics.predicted_rul.labels(unit_id=str(df["unit"].iloc[0])).set(
                result["predicted_rul"]
            )
            if result["status"] == "CRITICAL":
                metrics.failure_alerts_total.inc()
        except Exception:
            pass

        # Prediction log
        try:
            log_prediction({
                "endpoint": "upload",
                "unit_id": int(df["unit"].iloc[0]),
                "n_cycles": int(len(df)),
                "predicted_rul": result["predicted_rul"],
                "status": result["status"],
                "model": result["model_used"],
                "duration_ms": round(elapsed * 1000, 1),
            })
        except Exception:
            logger.exception("prediction_log_failed")

        return {
            "success": True,
            "unit_id": int(df["unit"].iloc[0]),
            "n_cycles": int(len(df)),
            "duration_ms": round(elapsed * 1000, 1),
            **result,
        }

    except HTTPException:
        try:
            metrics.predictions_total.labels(status="error").inc()
        except Exception:
            pass
        raise
    except Exception as exc:
        logger.exception("prediction_failed")
        try:
            metrics.predictions_total.labels(status="error").inc()
        except Exception:
            pass
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {type(exc).__name__}: {exc}",
        ) from exc


# --- Prediction: JSON (single cycle, requires history) ---
@app.post("/api/predict")
async def predict_json(request: PredictRequest):
    """
    Predict from a single cycle. Requires the client to have built up
    enough history — recommended to use /api/predict/upload instead.
    """
    started = time.perf_counter()

    try:
        from app.pipeline.schema import BASE_COLS, SENSOR_COLS

        row = {
            "unit": request.unit,
            "cycle": request.cycle,
            "op_setting_1": request.op_setting_1,
            "op_setting_2": request.op_setting_2,
            "op_setting_3": request.op_setting_3,
        }
        for i, col in enumerate(SENSOR_COLS):
            row[col] = request.sensors[i]

        df = pd.DataFrame([row])
        df = df[BASE_COLS]

        # We can't build rolling features from 1 row. Inflate by copying.
        # In production: client should upload history.
        df = pd.concat([df] * 10, ignore_index=True)
        df["cycle"] = range(1, 11)

        result = await run_in_threadpool(predict_from_history, df)
        elapsed = time.perf_counter() - started

        try:
            metrics.predictions_total.labels(status="success").inc()
            metrics.prediction_duration_seconds.observe(elapsed)
        except Exception:
            pass

        return {
            "success": True,
            "note": "Single-row input inflated to 10 cycles for feature engineering. "
                    "For accurate results, upload real history.",
            "duration_ms": round(elapsed * 1000, 1),
            **result,
        }

    except Exception as exc:
        logger.exception("predict_json_failed")
        try:
            metrics.predictions_total.labels(status="error").inc()
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}: {exc}",
        ) from exc