"""Tests for FastAPI endpoints."""

import io

from fastapi.testclient import TestClient


def test_root_returns_html(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert "Turbofan" in r.text


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "model_loaded" in body


def test_health_live(client: TestClient):
    r = client.get("/health/live")
    assert r.status_code == 200
    assert r.json() == {"status": "alive"}


def test_health_ready(client: TestClient):
    r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ready", "not_ready"}
    assert "checks" in body


def test_request_id_header(client: TestClient):
    r = client.get("/health")
    assert "x-request-id" in r.headers
    assert "x-process-time-ms" in r.headers


def test_metrics_endpoint(client: TestClient):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    assert "turbofan_http_requests_total" in r.text


def test_predict_upload_invalid_extension(client: TestClient):
    files = {"file": ("bad.json", io.BytesIO(b"{}"), "application/json")}
    r = client.post("/api/predict/upload", files=files)
    assert r.status_code == 400


def test_predict_upload_empty_file(client: TestClient):
    files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    r = client.post("/api/predict/upload", files=files)
    assert r.status_code == 400


def test_predict_upload_wrong_columns(client: TestClient):
    files = {"file": ("bad.txt", io.BytesIO(b"1 2 3 4\n5 6 7 8\n"), "text/plain")}
    r = client.post("/api/predict/upload", files=files)
    assert r.status_code == 400


def test_predict_json_invalid_payload(client: TestClient):
    r = client.post("/api/predict", json={})
    assert r.status_code == 422
