"""Optional CNN comparison: EfficientNetB0 transfer learning.

Run:  python ml/cnn/train_efficientnet.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml import config  # noqa: E402
from ml.models import build_cnn_classifier  # noqa: E402
from ml.training_utils import train_classifier  # noqa: E402


def main():
    config.set_seeds()
    model = build_cnn_classifier("efficientnet")
    train_classifier(model, name="efficientnet", best_path=config.EFFNET_MODEL_PATH,
                     frozen_layer_names=model.backbone_layer_names)


if __name__ == "__main__":
    main()
