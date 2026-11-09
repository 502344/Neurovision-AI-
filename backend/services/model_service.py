"""Loads (never trains) the saved models. Reports precisely what is missing."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field

import joblib
import numpy as np

from ml import config


class ModelNotAvailableError(RuntimeError):
    """Raised when required trained artefacts are missing."""


REQUIRED = {
    "CNN model (models/resnet50_best.keras)": config.CNN_MODEL_PATH,
    "ViT model (models/vit_best.keras)": config.VIT_MODEL_PATH,
    "XGBoost model (models/xgboost_asd.json)": config.XGB_MODEL_PATH,
    "Selected features (features/selected_features.npy)": config.SELECTED_IDX_PATH,
    "Pipeline metadata (models/pipeline_meta.json)": config.PIPELINE_META_PATH,
}


def missing_artifacts() -> list[str]:
    return [label for label, p in REQUIRED.items() if not p.exists()]


@dataclass
class ModelBundle:
    meta: dict
    gradcam: object
    vit_extractor: object
    xgb: object
    selected: np.ndarray
    severity: object | None = None
    svm: object | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_bundle: ModelBundle | None = None
_load_lock = threading.Lock()


def model_status() -> dict:
    miss = missing_artifacts()
    sev = config.SEVERITY_MODEL_PATH.exists()
    return {"available": not miss, "missing": miss,
            "message": None if not miss else config.MODEL_MISSING_MESSAGE,
            "status_banner": None if not miss else
            "MODEL STATUS: Training dataset required. No experimental performance values are displayed "
            "until real labelled data are supplied and evaluated.",
            "severity_model_available": sev, "loaded": _bundle is not None}


def get_bundle(force_reload: bool = False) -> ModelBundle:
    """Load models once. Raises ModelNotAvailableError if training has not been completed."""
    global _bundle
    with _load_lock:
        if _bundle is not None and not force_reload:
            return _bundle
        if missing_artifacts():
            raise ModelNotAvailableError(config.MODEL_MISSING_MESSAGE)
        from xgboost import XGBClassifier
        from ml.explainability.gradcam import GradCamExtractor
        from ml.models import feature_extractor, load_keras_model

        meta = json.loads(config.PIPELINE_META_PATH.read_text())
        if int(meta["image_size"]) != config.IMAGE_SIZE:
            raise ModelNotAvailableError(
                f"Models were trained with IMAGE_SIZE={meta['image_size']} but the current configuration is "
                f"{config.IMAGE_SIZE}. Set NEUROVISION_IMAGE_SIZE={meta['image_size']}.")
        cnn = load_keras_model(config.CNN_MODEL_PATH)
        vit = load_keras_model(config.VIT_MODEL_PATH)
        xgb = XGBClassifier(); xgb.load_model(str(config.XGB_MODEL_PATH))
        sev = None
        if config.SEVERITY_MODEL_PATH.exists():
            sev = XGBClassifier(); sev.load_model(str(config.SEVERITY_MODEL_PATH))
        svm = joblib.load(config.SVM_MODEL_PATH) if config.SVM_MODEL_PATH.exists() else None
        _bundle = ModelBundle(
            meta=meta, gradcam=GradCamExtractor(cnn),
            vit_extractor=feature_extractor(vit, config.VIT_FEATURE_LAYER),
            xgb=xgb, selected=np.load(config.SELECTED_IDX_PATH), severity=sev, svm=svm)
        return _bundle


def reset_bundle() -> None:
    global _bundle
    _bundle = None


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None
