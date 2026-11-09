"""Shared feature-extraction driver (manifest order -> .npy + metadata CSV)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml import config
from ml.data import make_dataset
from ml.models import feature_extractor, load_keras_model
from ml.preprocessing.dataset_validation import load_manifest


def extract_features(model_path: Path, layer_name: str, out_name: str) -> np.ndarray:
    config.ensure_dirs()
    config.set_seeds()
    if not Path(model_path).exists():
        print(f"[extract] Trained model not found: {model_path}. Train it first.")
        sys.exit(1)
    try:
        manifest = load_manifest()
    except FileNotFoundError as exc:
        print(f"[extract] {exc}")
        sys.exit(1)
    model = load_keras_model(model_path)
    extractor = feature_extractor(model, layer_name)
    ds = make_dataset(manifest["image_path"], None, batch_size=config.BATCH_SIZE, cache=False)
    feats = extractor.predict(ds, verbose=0).astype(np.float32)
    if len(feats) != len(manifest):
        raise RuntimeError("Feature count does not match manifest length.")
    np.save(config.FEATURES_DIR / f"{out_name}.npy", feats)
    meta = manifest[["image_path", "patient_id", "label", "label_idx", "severity", "split"]].copy()
    meta.insert(0, "row", range(len(meta)))
    meta.to_csv(config.FEATURES_DIR / "feature_metadata.csv", index=False)
    print(f"[extract] {out_name}: shape={feats.shape} from {Path(model_path).name} (layer '{layer_name}')")
    return feats


def load_feature_table():
    """Return (metadata DataFrame, dict of available feature arrays)."""
    meta_path = config.FEATURES_DIR / "feature_metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError("features/feature_metadata.csv missing. Run the feature extraction scripts.")
    return pd.read_csv(meta_path)
