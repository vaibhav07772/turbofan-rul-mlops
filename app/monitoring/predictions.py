"""Prediction audit logging to disk (JSON Lines).

Used for offline analysis and drift detection.
Never blocks requests.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger("predictions")

_LOCK = threading.Lock()
_LOG_PATH: Path | None = None


def _get_log_path() -> Path:
    global _LOG_PATH
    if _LOG_PATH is None:
        settings = get_settings()
        log_dir = settings.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        _LOG_PATH = log_dir / "predictions.jsonl"
    return _LOG_PATH


def log_prediction(record: dict) -> None:
    """Append one prediction record to the prediction log."""
    try:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        line = json.dumps(payload, default=str) + "\n"
        path = _get_log_path()
        with _LOCK, path.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        logger.exception("prediction_log_failed")