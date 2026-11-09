"""Extract deep features from the trained CNN (GAP layer, before the classification head).

Run:  python ml/cnn/extract_cnn_features.py
Saves features/cnn_features.npy and features/feature_metadata.csv (row order == split manifest).

Caveat: features for TRAIN images come from a network that saw those images during training and may be
optimistic relative to unseen images. Downstream models are therefore selected on the validation split only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml import config  # noqa: E402
from ml.features_common import extract_features  # noqa: E402

if __name__ == "__main__":
    extract_features(config.CNN_MODEL_PATH, config.CNN_FEATURE_DIM_LAYER, "cnn_features")
