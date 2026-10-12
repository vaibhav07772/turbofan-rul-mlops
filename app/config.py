"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "Turbofan RUL MLOps"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    log_json: bool = False

    # MLflow (SQLite backend for cross-platform model registry)
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_artifact_location: str = "./mlruns"
    mlflow_experiment_name: str = "turbofan-rul"
    mlflow_registered_model_name: str = "turbofan-rul-model"
    mlflow_auto_register: bool = True

    # Uploads
    max_upload_bytes: int = 50 * 1024 * 1024  # 50 MB
    allowed_extensions: str = ".csv,.txt"

    # Paths
    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def extension_set(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_extensions.split(",") if e.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
