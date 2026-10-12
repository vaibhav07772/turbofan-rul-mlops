"""FastAPI app for Turbofan RUL prediction."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

from app.config import get_settings
from app.logging_config import configure_logging, get_logger
from app.middleware import RequestContextMiddleware
from app.monitoring import metrics

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


# --- Prometheus metrics ---
@app.get("/metrics")
async def prometheus_metrics():
    payload, content_type = metrics.render_metrics()
    return Response(content=payload, media_type=content_type)


# --- Health probes ---
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }


@app.get("/health/live")
async def health_live():
    return {"status": "alive"}


@app.get("/health/ready")
async def health_ready():
    checks = {
        "data_dir": settings.data_dir.exists(),
        "model_dir": settings.model_dir.exists(),
    }
    ready = all(checks.values())
    return {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
    }