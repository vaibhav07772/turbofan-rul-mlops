"""Quick test: does DagsHub MLflow connection work?"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.mlops.mlflow_utils import setup_mlflow
import mlflow

settings = get_settings()

print("=" * 55)
print(" DagsHub MLflow Connection Test")
print("=" * 55)
print(f"\nTracking URI: {settings.mlflow_tracking_uri}")
print(f"Username: {settings.mlflow_tracking_username or '(not set)'}")
print(f"Password: {'*' * 20 if settings.mlflow_tracking_password else '(not set)'}")
print(f"Remote mode: {settings.is_remote_mlflow}")

if not settings.mlflow_tracking_username or not settings.mlflow_tracking_password:
    print("\n[!] Username/Password not set in .env")
    sys.exit(1)

setup_mlflow()

try:
    with mlflow.start_run(run_name="dagshub-connection-test") as run:
        mlflow.log_param("test", "connection")
        mlflow.log_metric("test_metric", 1.0)
        print(f"\n[✓] Connection successful!")
        print(f"    Run ID: {run.info.run_id}")
        print(f"    View at: https://dagshub.com/vaibhav07772/turbofan-rul-mlops/experiments")
except Exception as exc:
    print(f"\n[✗] Connection failed: {type(exc).__name__}: {exc}")
    sys.exit(1)