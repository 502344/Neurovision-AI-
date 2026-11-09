"""Executed in a subprocess with the smoke environment: exercises the real API against real saved models."""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from tests.conftest import make_fundus_like

c = TestClient(app)
out = {}
h = c.get("/health").json(); out["health_models"] = h["models_available"]
p = Path(sys.argv[1]); make_fundus_like(p, 4242)
with open(p, "rb") as f:
    r = c.post("/predict", files={"file": ("x.png", f, "image/png")})
out["status"] = r.status_code
body = r.json()
out["keys_ok"] = all(k in body for k in ("prediction", "confidence", "severity", "severity_available", "gradcam_path", "model", "processing_time"))
out["report_status"] = c.get(body["report_url"]).status_code
out["overlay_status"] = c.get(body["gradcam_url"]).status_code
out["metrics"] = c.get("/metrics").json()["available"]
out["model_info"] = c.get("/model-info").json()["available"]
out["dataset"] = c.get("/dataset-info").json()["available"]
out["plot"] = c.get("/plots/roc_curve.png").status_code
buf = io.BytesIO(); Image.new("RGB", (100, 100)).save(buf, "PNG")
out["validate"] = c.post("/validate-image", files={"file": ("a.png", buf.getvalue(), "image/png")}).status_code
print("JSON>>" + json.dumps(out))
