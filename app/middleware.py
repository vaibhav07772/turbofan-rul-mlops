"""Request middleware: request ID, timing, and Prometheus metrics."""

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.monitoring import metrics


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request ID, measure duration, and record metrics."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        logger = structlog.get_logger("request")
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_failed")
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = f"{elapsed_ms:.1f}"

        # Skip /metrics to avoid recursion
        if request.url.path != "/metrics":
            try:
                metrics.http_requests_total.labels(
                    method=request.method,
                    path=request.url.path,
                    status=str(response.status_code),
                ).inc()
                metrics.http_request_duration_seconds.labels(
                    method=request.method,
                    path=request.url.path,
                ).observe(elapsed_ms / 1000.0)
            except Exception:
                pass

        logger.info(
            "request_completed",
            status_code=response.status_code,
            duration_ms=round(elapsed_ms, 1),
        )
        return response