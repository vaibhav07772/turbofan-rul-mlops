"""Prometheus metrics for Turbofan RUL MLOps."""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# --- HTTP / API ---
http_requests_total = Counter(
    "turbofan_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

http_request_duration_seconds = Histogram(
    "turbofan_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# --- Predictions ---
predictions_total = Counter(
    "turbofan_predictions_total",
    "Total predictions",
    ["status"],  # success / error
)

predicted_rul = Gauge(
    "turbofan_predicted_rul",
    "Latest predicted RUL (cycles)",
    ["unit_id"],
)

prediction_duration_seconds = Histogram(
    "turbofan_prediction_duration_seconds",
    "Prediction latency",
    buckets=(0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
)

# --- Training ---
training_runs_total = Counter(
    "turbofan_training_runs_total",
    "Total training runs",
    ["best_model"],
)

training_duration_seconds = Histogram(
    "turbofan_training_duration_seconds",
    "Training duration",
    ["model"],
    buckets=(0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)

# --- Alerts ---
failure_alerts_total = Counter(
    "turbofan_failure_alerts_total",
    "Total critical RUL alerts (< threshold)",
)


def render_metrics() -> tuple[bytes, str]:
    """Return (payload, content_type) for /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
