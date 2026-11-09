"""Extract [CLS]-token features by actually running the Vision Transformer.

Run:  python ml/vit/extract_vit_features.py
Saves features/vit_features.npy
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml import config  # noqa: E402
from ml.features_common import extract_features  # noqa: E402

if __name__ == "__main__":
    extract_features(config.VIT_MODEL_PATH, config.VIT_FEATURE_LAYER, "vit_features")
