"""Central configuration for NeuroVision AI.

Every value can be overridden with an environment variable named
``NEUROVISION_<NAME>`` (e.g. ``NEUROVISION_EPOCHS=5``). This keeps Colab runs,
local runs and automated tests reproducible without editing source files.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


def _env(name: str, default, cast=str):
    raw = os.environ.get(f"NEUROVISION_{name}")
    if raw is None:
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "y"}
    return cast(raw)


# ----------------------------------------------------------------- paths
ROOT_DIR = Path(_env("ROOT", Path(__file__).resolve().parent.parent, Path))
DATASET_DIR = ROOT_DIR / "dataset"
MODELS_DIR = ROOT_DIR / "models"
FEATURES_DIR = ROOT_DIR / "features"
RESULTS_DIR = ROOT_DIR / "results"
REPORTS_DIR = ROOT_DIR / "reports"
UPLOADS_DIR = ROOT_DIR / "uploads"
GRADCAM_DIR = RESULTS_DIR / "gradcam"

# Optional CSV dataset (Format B): image_path,patient_id,label,severity
DATASET_CSV = _env("DATASET_CSV", "", str)          # empty -> use folder format
DATASET_ROOT = _env("DATASET_ROOT", "", str)        # base dir for relative CSV paths
SPLIT_MANIFEST = RESULTS_DIR / "split_manifest.csv"  # written by dataset_validation

# ----------------------------------------------------------- core settings
IMAGE_SIZE = _env("IMAGE_SIZE", 224, int)
BATCH_SIZE = _env("BATCH_SIZE", 16, int)
EPOCHS = _env("EPOCHS", 30, int)
FINE_TUNE_EPOCHS = _env("FINE_TUNE_EPOCHS", 10, int)      # 0 disables fine-tuning
FINE_TUNE_LAST_LAYERS = _env("FINE_TUNE_LAST_LAYERS", 30, int)
LEARNING_RATE = _env("LEARNING_RATE", 1e-4, float)
FINE_TUNE_LEARNING_RATE = _env("FINE_TUNE_LEARNING_RATE", 1e-5, float)
RANDOM_STATE = _env("RANDOM_STATE", 42, int)
TOP_K_FEATURES = _env("TOP_K_FEATURES", 500, int)
AUTO_TOPK = _env("AUTO_TOPK", True, bool)   # pick K by validation ROC-AUC from TOP_K_CANDIDATES; False -> TOP_K_FEATURES
RF_N_ESTIMATORS = _env("RF_N_ESTIMATORS", 500, int)
TOP_K_CANDIDATES = [int(k) for k in _env("TOP_K_CANDIDATES", "50,100,250,500,1000", str).split(",")]

# --------------------------------------------------------------- data split
TRAIN_FRACTION = _env("TRAIN_FRACTION", 0.70, float)
VAL_FRACTION = _env("VAL_FRACTION", 0.15, float)
TEST_FRACTION = _env("TEST_FRACTION", 0.15, float)
MAX_FILE_SIZE_MB = _env("MAX_FILE_SIZE_MB", 15, int)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/tiff"}

# ------------------------------------------------------------ preprocessing
CROP_BLACK_BORDER = _env("CROP_BLACK_BORDER", True, bool)
BLACK_THRESHOLD = _env("BLACK_THRESHOLD", 12, int)
MIN_IMAGE_SIDE = _env("MIN_IMAGE_SIDE", 64, int)
AUGMENT_HORIZONTAL_FLIP = _env("AUGMENT_HORIZONTAL_FLIP", True, bool)

# ------------------------------------------------------------------- models
CNN_BACKBONE = _env("CNN_BACKBONE", "resnet50", str)     # resnet50 | efficientnet
USE_PRETRAINED = _env("PRETRAINED", True, bool)          # ImageNet weights (needs internet)
CNN_FEATURE_DIM_LAYER = "cnn_features"                   # penultimate GAP layer name
VIT_FEATURE_LAYER = "vit_features"
VIT_PRESET = _env("VIT_PRESET", "vit_base_patch16_224_imagenet", str)  # "" -> random init
# Used only when VIT_PRESET is empty (unit tests / smoke tests -- NOT for research runs)
VIT_TINY = dict(patch_size=16, num_layers=2, num_heads=2, hidden_dim=64, mlp_dim=128)
DENSE_UNITS = _env("DENSE_UNITS", 128, int)
DROPOUT = _env("DROPOUT", 0.3, float)

FUSION_METHOD = _env("FUSION_METHOD", "concat", str)     # concat | l2_concat | weighted_concat
FUSION_CNN_WEIGHT = _env("FUSION_CNN_WEIGHT", 0.5, float)

CLASS_NAMES = ["Non_ASD", "ASD"]                         # index 1 = ASD (positive class)
SEVERITY_CLASSES = ["Mild", "Moderate", "Severe"]
DECISION_THRESHOLD = _env("DECISION_THRESHOLD", 0.5, float)
MIN_SEVERITY_SAMPLES_PER_CLASS = _env("MIN_SEVERITY_SAMPLES_PER_CLASS", 5, int)

# ----------------------------------------------------------------- artefacts
CNN_MODEL_PATH = MODELS_DIR / "resnet50_best.keras"
EFFNET_MODEL_PATH = MODELS_DIR / "efficientnet_best.keras"
VIT_MODEL_PATH = MODELS_DIR / "vit_best.keras"
XGB_MODEL_PATH = MODELS_DIR / "xgboost_asd.json"
XGB_FULL_MODEL_PATH = MODELS_DIR / "xgboost_fused_full.json"
SVM_MODEL_PATH = MODELS_DIR / "svm_asd.joblib"
SEVERITY_MODEL_PATH = MODELS_DIR / "xgboost_severity.json"
PIPELINE_META_PATH = MODELS_DIR / "pipeline_meta.json"
SELECTED_IDX_PATH = FEATURES_DIR / "selected_features.npy"
FEATURE_IMPORTANCE_CSV = RESULTS_DIR / "feature_importance.csv"

MEDICAL_DISCLAIMER = (
    "This is an AI-assisted research screening prototype and is not a clinical diagnosis. "
    "Results should not be used as a substitute for assessment by qualified healthcare professionals."
)
MODEL_DISPLAY_NAME = "CNN + ViT + Random Forest + XGBoost"
MODEL_MISSING_MESSAGE = (
    "The trained model is not available. Please complete model training before using prediction."
)


def ensure_dirs() -> None:
    for d in (MODELS_DIR, FEATURES_DIR, RESULTS_DIR, REPORTS_DIR, UPLOADS_DIR, GRADCAM_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int | None = None) -> int:
    """Seed python, numpy and (when installed) TensorFlow for reproducibility."""
    seed = RANDOM_STATE if seed is None else seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except Exception:  # TensorFlow not installed (e.g. lightweight API tests)
        pass
    return seed


def as_dict() -> dict:
    """Serialisable snapshot of the configuration (saved with each experiment)."""
    keys = ["IMAGE_SIZE", "BATCH_SIZE", "EPOCHS", "FINE_TUNE_EPOCHS", "FINE_TUNE_LAST_LAYERS",
            "LEARNING_RATE", "FINE_TUNE_LEARNING_RATE", "RANDOM_STATE", "TOP_K_FEATURES",
            "TOP_K_CANDIDATES", "AUTO_TOPK", "RF_N_ESTIMATORS", "TRAIN_FRACTION", "VAL_FRACTION", "TEST_FRACTION",
            "CROP_BLACK_BORDER", "CNN_BACKBONE", "USE_PRETRAINED", "VIT_PRESET", "DENSE_UNITS",
            "DROPOUT", "FUSION_METHOD", "FUSION_CNN_WEIGHT", "DECISION_THRESHOLD"]
    g = globals()
    return {k: g[k] for k in keys}
