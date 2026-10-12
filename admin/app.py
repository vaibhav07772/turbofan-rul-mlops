"""Turbofan RUL — Streamlit Admin Dashboard.

Monitors MLflow runs, live predictions, and model health.

Usage:
    streamlit run admin/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st
from mlflow.tracking import MlflowClient

# --- Config ---
ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"
PRED_LOG = ROOT / "data" / "logs" / "predictions.jsonl"
MODEL_DIR = ROOT / "data" / "models"
EXPERIMENT = "turbofan-rul"
REGISTERED_MODEL = "turbofan-rul-model"

st.set_page_config(
    page_title="Turbofan RUL Admin",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Helpers ---
@st.cache_resource
def _client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_URI)
    return MlflowClient()


@st.cache_data(ttl=10)
def load_runs(limit: int = 100) -> pd.DataFrame:
    """Load top-level parent runs (excludes nested)."""
    client = _client()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return pd.DataFrame()

    all_runs = client.search_runs(
        [exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=limit * 3,
    )
    parent_runs = [r for r in all_runs if r.data.tags.get("mlflow.parentRunId") is None]

    rows = []
    for r in parent_runs[:limit]:
        start = r.info.start_time
        end = r.info.end_time
        rows.append({
            "run_id": r.info.run_id,
            "name": r.data.tags.get("mlflow.runName", ""),
            "status": r.info.status,
            "best_model": r.data.tags.get("best_model", ""),
            "n_train": r.data.params.get("n_train"),
            "n_val": r.data.params.get("n_val"),
            "n_features": r.data.params.get("n_features"),
            "rmse": r.data.metrics.get("best_rmse"),
            "mae": r.data.metrics.get("best_mae"),
            "r2": r.data.metrics.get("best_r2"),
            "start_time": pd.to_datetime(start, unit="ms") if start else None,
            "duration_s": round((end - start) / 1000, 2) if (start and end) else None,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=10)
def load_registry() -> pd.DataFrame:
    client = _client()
    try:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    except Exception:
        return pd.DataFrame()
    rows = []
    for v in versions:
        rows.append({
            "version": int(v.version),
            "stage": v.current_stage,
            "status": v.status,
            "created": pd.to_datetime(v.creation_timestamp, unit="ms"),
            "run_id": v.run_id,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("version", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(ttl=5)
def load_predictions() -> pd.DataFrame:
    if not PRED_LOG.exists():
        return pd.DataFrame()
    records = []
    with PRED_LOG.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(ttl=30)
def model_info() -> dict:
    info = {"exists": False}
    model_file = MODEL_DIR / "model.pkl"
    feature_file = MODEL_DIR / "feature_columns.pkl"
    if model_file.exists():
        info["exists"] = True
        info["size_kb"] = round(model_file.stat().st_size / 1024, 1)
        info["modified"] = pd.to_datetime(model_file.stat().st_mtime, unit="s")
    if feature_file.exists():
        try:
            import joblib
            features = joblib.load(feature_file)
            info["n_features"] = len(features)
        except Exception:
            pass
    return info


# --- Sidebar ---
with st.sidebar:
    st.title("✈️ Turbofan RUL Admin")
    st.caption("MLOps monitoring for predictive maintenance")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Overview", "Experiments", "Predictions", "Model Health"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("**Tracking URI**")
    st.code(MLFLOW_URI, language="text")

    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# --- Load data ---
try:
    runs_df = load_runs()
    registry_df = load_registry()
    preds_df = load_predictions()
    info = model_info()
    load_error = None
except Exception as exc:
    runs_df = pd.DataFrame()
    registry_df = pd.DataFrame()
    preds_df = pd.DataFrame()
    info = {}
    load_error = f"{type(exc).__name__}: {exc}"

if load_error:
    st.error(f"⚠️ Data load failed: {load_error}")
    st.info("Run a training first: `python scripts/train.py`")
    st.stop()


# --- Pages ---
if page == "Overview":
    st.title("📊 Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total runs", len(runs_df))

    best_rmse = (
        f"{runs_df['rmse'].min():.3f}"
        if not runs_df.empty and runs_df["rmse"].notna().any()
        else "—"
    )
    col2.metric("Best RMSE", best_rmse)
    col3.metric("Model versions", len(registry_df))
    col4.metric("Total predictions", len(preds_df))

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Latest training run")
        if not runs_df.empty:
            latest = runs_df.iloc[0]
            st.markdown(f"**Run:** `{latest['name']}`")
            st.markdown(f"**Best model:** `{latest['best_model']}`")
            st.markdown(f"**Status:** {latest['status']}")
            st.markdown(f"**RMSE:** {latest['rmse']}")
            st.markdown(f"**MAE:** {latest['mae']}")
            st.markdown(f"**R²:** {latest['r2']}")
            st.markdown(f"**Duration:** {latest['duration_s']}s")
            st.markdown(f"**Train / Val:** {latest['n_train']} / {latest['n_val']}")
            st.markdown(f"**Features:** {latest['n_features']}")
        else:
            st.info("No runs yet. Run `python scripts/train.py`.")

    with right:
        st.subheader("Local model")
        if info.get("exists"):
            st.success("✅ Model loaded from disk")
            st.markdown(f"**Size:** {info.get('size_kb')} KB")
            st.markdown(f"**Features:** {info.get('n_features')}")
            st.markdown(f"**Modified:** {info.get('modified')}")
        else:
            st.warning("No model file found. Train first.")


elif page == "Experiments":
    st.title("🧪 Experiments")
    st.caption(f"Experiment: `{EXPERIMENT}`")

    if runs_df.empty:
        st.info("No runs yet.")
    else:
        chart_df = runs_df.dropna(subset=["rmse"])
        if not chart_df.empty:
            fig = px.bar(
                chart_df,
                x="name",
                y="rmse",
                color="best_model",
                title="RMSE across recent runs (lower is better)",
                labels={"rmse": "RMSE (cycles)", "name": "Run name"},
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("All runs")
        display = runs_df[[
            "name", "best_model", "status",
            "rmse", "mae", "r2",
            "duration_s", "start_time",
        ]].copy()
        st.dataframe(display, use_container_width=True, hide_index=True)


elif page == "Predictions":
    st.title("🔮 Predictions")

    if preds_df.empty:
        st.info("No predictions logged yet. Hit `/api/predict/upload`.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total predictions", len(preds_df))

        if "status" in preds_df.columns:
            critical = (preds_df["status"] == "CRITICAL").sum()
            warning = (preds_df["status"] == "WARNING").sum()
            col2.metric("⚠️ Critical alerts", int(critical))
            col3.metric("🟡 Warnings", int(warning))

        st.divider()

        if "timestamp" in preds_df.columns and "predicted_rul" in preds_df.columns:
            timeline = preds_df.sort_values("timestamp")
            fig = px.line(
                timeline,
                x="timestamp",
                y="predicted_rul",
                markers=True,
                color="status",
                title="Predicted RUL over time",
                labels={"predicted_rul": "RUL (cycles)", "timestamp": "Time"},
            )
            fig.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="CRITICAL threshold")
            fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="WARNING threshold")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent predictions")
        cols = [c for c in [
            "timestamp", "unit_id", "predicted_rul", "status",
            "model", "n_cycles", "duration_ms",
        ] if c in preds_df.columns]
        st.dataframe(
            preds_df.sort_values("timestamp", ascending=False)[cols].head(100),
            use_container_width=True,
            hide_index=True,
        )


elif page == "Model Health":
    st.title("🏥 Model Health")

    if preds_df.empty:
        st.info("No predictions yet.")
    else:
        st.subheader("Status distribution")
        if "status" in preds_df.columns:
            status_counts = preds_df["status"].value_counts().reset_index()
            status_counts.columns = ["status", "count"]
            fig = px.pie(
                status_counts,
                names="status",
                values="count",
                color="status",
                color_discrete_map={
                    "HEALTHY": "green",
                    "WARNING": "orange",
                    "CRITICAL": "red",
                },
                title="Prediction status distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("RUL distribution")
        if "predicted_rul" in preds_df.columns:
            fig = px.histogram(
                preds_df,
                x="predicted_rul",
                nbins=20,
                title="Predicted RUL distribution",
                labels={"predicted_rul": "Predicted RUL (cycles)"},
            )
            st.plotly_chart(fig, use_container_width=True)

            stats = preds_df["predicted_rul"].describe()
            st.markdown("**Statistics**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean RUL", f"{stats['mean']:.1f}")
            c2.metric("Median RUL", f"{stats['50%']:.1f}")
            c3.metric("Min RUL", f"{stats['min']:.1f}")
            c4.metric("Max RUL", f"{stats['max']:.1f}")


st.sidebar.divider()
st.sidebar.caption("Part of **Turbofan RUL MLOps**")
st.sidebar.caption("Built by **Vaibhav Singh Rajput** 🚀")