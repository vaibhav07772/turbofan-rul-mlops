"""Turbofan RUL MLOps — application entry point."""

import os
import sys
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.logging_config import configure_logging, get_logger  # noqa: E402

configure_logging()
logger = get_logger("run")


def main() -> None:
    import uvicorn

    print("=" * 50)
    print(" Turbofan RUL MLOps")
    print("=" * 50)

    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"[run] Starting Uvicorn on {host}:{port}")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()