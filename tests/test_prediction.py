import json
import subprocess
import sys
from pathlib import Path

import pytest

from ml.preprocessing.image_preprocessing import InvalidImageError

ROOT = Path(__file__).resolve().parents[1]


def test_prediction_refuses_without_trained_models(sample_image):
    from backend.services.model_service import ModelNotAvailableError, missing_artifacts
    from backend.services.prediction_service import predict_retinal_image
    if not missing_artifacts():
        pytest.skip("Trained models are present")
    with pytest.raises(ModelNotAvailableError):
        predict_retinal_image(sample_image)


def test_prediction_rejects_invalid_image(tmp_path):
    from backend.services.prediction_service import predict_retinal_image
    bad = tmp_path / "bad.png"; bad.write_bytes(b"nope")
    with pytest.raises(InvalidImageError):
        predict_retinal_image(bad)


def test_prediction(smoke_env, tmp_path):
    """Full pipeline on a synthetic image with tiny random-weight models: checks structure and wiring only."""
    from tests.conftest import make_fundus_like
    img = tmp_path / "q.png"; make_fundus_like(img, 999)
    code = f"""
import json
from backend.services.prediction_service import predict_retinal_image
r = predict_retinal_image({str(img)!r})
r = {{k: v for k, v in r.items() if not k.endswith('_b64')}}
print('JSON>>' + json.dumps(r))
"""
    r = subprocess.run([sys.executable, "-c", code], env=smoke_env["env"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(r.stdout.split("JSON>>")[1])
    assert res["prediction"] in ("ASD", "Non-ASD")
    assert 0.5 <= res["confidence"] <= 1.0 and 0.0 <= res["probability_asd"] <= 1.0
    assert res["model"] == "CNN + ViT + Random Forest + XGBoost" and res["processing_time"] > 0
    assert Path(res["gradcam_path"]).exists() and Path(res["report_path"]).read_bytes()[:4] == b"%PDF"
    assert res["severity"] in (None, "Mild", "Moderate", "Severe")
    assert "not a clinical diagnosis" in res["disclaimer"]
