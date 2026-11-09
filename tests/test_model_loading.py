import os

import numpy as np
import pytest

from backend.services.model_service import ModelNotAvailableError, get_bundle, missing_artifacts, model_status


def test_model_loading_reports_missing_models():
    # The repository ships without trained models: nothing may be silently faked.
    if not missing_artifacts():
        pytest.skip("Trained models are present in this checkout")
    assert not model_status()["available"]
    with pytest.raises(ModelNotAvailableError, match="trained model is not available"):
        get_bundle()


def test_keras_roundtrip_with_custom_layers(tmp_path):
    os.environ.setdefault("NEUROVISION_PRETRAINED", "0")
    from ml.models import build_cnn_classifier, build_vit_classifier, load_keras_model
    m = build_cnn_classifier("resnet50", pretrained=False, image_size=64)
    p = tmp_path / "cnn.keras"; m.save(p)
    m2 = load_keras_model(p)
    x = np.random.rand(2, 64, 64, 3).astype("float32") * 255
    np.testing.assert_allclose(m.predict(x, verbose=0), m2.predict(x, verbose=0), rtol=1e-4, atol=1e-5)
    from ml import config
    saved = config.VIT_PRESET
    config.VIT_PRESET = ""
    try:
        v = build_vit_classifier(pretrained=False, image_size=64)
    finally:
        config.VIT_PRESET = saved
    pv = tmp_path / "vit.keras"; v.save(pv)
    v2 = load_keras_model(pv)
    np.testing.assert_allclose(v.predict(x, verbose=0), v2.predict(x, verbose=0), rtol=1e-4, atol=1e-5)
