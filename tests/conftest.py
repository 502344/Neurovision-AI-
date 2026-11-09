import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_fundus_like(path: Path, seed: int, size: int = 96, ext_border: bool = True):
    """SYNTHETIC test image (disc on black background). Used ONLY to test code plumbing, never for results."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    r = size * 0.42
    mask = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < r ** 2
    img = np.zeros((size, size, 3), np.uint8)
    base = np.array([140 + rng.integers(-30, 30), 70 + rng.integers(-20, 20), 40], np.float32)
    noise = rng.normal(0, 12, (size, size, 3))
    img[mask] = np.clip(base + noise[mask], 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


@pytest.fixture()
def sample_image(tmp_path):
    p = tmp_path / "eye.png"
    make_fundus_like(p, 1)
    return p


@pytest.fixture(scope="session")
def smoke_env(tmp_path_factory):
    """Runs the ENTIRE pipeline on synthetic data with tiny random-weight models in an isolated directory.

    This verifies wiring (files, shapes, saving/loading, API) -- the resulting numbers are meaningless and are
    never surfaced anywhere outside the temporary test directory.
    """
    root = tmp_path_factory.mktemp("nv_smoke")
    (root / "images").mkdir()
    rows = []
    for i in range(60):
        label = "ASD" if i % 2 == 0 else "Non-ASD"
        pid = f"P{i // 3:03d}" if False else f"P{i:03d}"
        make_fundus_like(root / "images" / f"img{i:03d}.png", i)
        sev = ["Mild", "Moderate", "Severe"][(i // 2) % 3] if label == "ASD" else ""
        rows.append({"image_path": f"images/img{i:03d}.png", "patient_id": pid, "label": label, "severity": sev})
    pd.DataFrame(rows).to_csv(root / "labels.csv", index=False)
    env = {**os.environ, "NEUROVISION_ROOT": str(root), "NEUROVISION_IMAGE_SIZE": "64", "NEUROVISION_PRETRAINED": "0",
           "NEUROVISION_VIT_PRESET": "", "NEUROVISION_EPOCHS": "1", "NEUROVISION_FINE_TUNE_EPOCHS": "0",
           "NEUROVISION_BATCH_SIZE": "8", "NEUROVISION_TOP_K_CANDIDATES": "10,20", "NEUROVISION_TOP_K_FEATURES": "20",
           "NEUROVISION_RF_N_ESTIMATORS": "20", "TF_CPP_MIN_LOG_LEVEL": "3", "NEUROVISION_MIN_SEVERITY_SAMPLES_PER_CLASS": "3"}
    steps = [["ml/preprocessing/dataset_validation.py", "--csv", str(root / "labels.csv")],
             ["ml/cnn/train_resnet.py"], ["ml/vit/train_vit.py"], ["ml/cnn/extract_cnn_features.py"],
             ["ml/vit/extract_vit_features.py"], ["ml/fusion/feature_fusion.py"],
             ["ml/feature_selection/random_forest_selection.py"], ["ml/classification/train_xgboost.py"],
             ["ml/classification/train_svm.py"], ["ml/severity/train_severity_xgboost.py"],
             ["ml/evaluation/evaluate_models.py"]]
    for s in steps:
        r = subprocess.run([sys.executable, str(ROOT / s[0]), *s[1:]], env=env, cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, f"{s[0]} failed:\n{r.stdout[-1500:]}\n{r.stderr[-2500:]}"
    return {"root": root, "env": env}
