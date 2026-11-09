import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def _png():
    b = io.BytesIO(); Image.new("RGB", (100, 100), (120, 60, 30)).save(b, "PNG"); return b.getvalue()


def test_api_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert client.get("/").json()["name"] == "NEUROVISION AI"


def test_api_rejects_bad_extension_mime_and_corrupt():
    assert client.post("/predict", files={"file": ("a.gif", _png(), "image/gif")}).status_code == 415
    assert client.post("/predict", files={"file": ("a.png", _png(), "text/plain")}).status_code == 415
    assert client.post("/predict", files={"file": ("a.png", b"not-an-image", "image/png")}).status_code == 422
    assert client.post("/predict", files={"file": ("a.png", b"", "image/png")}).status_code == 400


def test_api_rejects_oversize(monkeypatch):
    from ml import config
    monkeypatch.setattr(config, "MAX_FILE_SIZE_MB", 0.0001)
    assert client.post("/predict", files={"file": ("a.png", _png(), "image/png")}).status_code == 413


def test_api_prediction_without_models_returns_503_not_fake_result():
    from backend.services.model_service import missing_artifacts
    if not missing_artifacts():
        pytest.skip("Trained models are present")
    r = client.post("/predict", files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 503 and "trained model is not available" in r.json()["detail"]
    m = client.get("/metrics").json()
    assert m["available"] is False and m["comparison"] == []


def test_api_prediction(smoke_env, tmp_path):
    """Real API + real (tiny, synthetic-data) saved models in an isolated workspace."""
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "_smoke_api_check.py"), str(tmp_path / "u.png")],
                       env=smoke_env["env"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    o = json.loads(r.stdout.split("JSON>>")[1])
    assert o["health_models"] and o["status"] == 200 and o["keys_ok"]
    assert o["report_status"] == 200 and o["overlay_status"] == 200
    assert o["metrics"] and o["model_info"] and o["dataset"] and o["plot"] == 200 and o["validate"] == 200
