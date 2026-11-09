"""Train the CNN baseline: ResNet50 -> GAP -> Dense -> ASD / Non-ASD.

Run:  python ml/cnn/train_resnet.py
Saves models/resnet50_best.keras, results/resnet50_history.json, results/training_history.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml import config  # noqa: E402
from ml.models import build_cnn_classifier  # noqa: E402
from ml.training_utils import train_classifier  # noqa: E402


def main():
    config.set_seeds()
    model = build_cnn_classifier("resnet50")
    train_classifier(model, name="resnet50", best_path=config.CNN_MODEL_PATH,
                     frozen_layer_names=model.backbone_layer_names, generic_plot="training_history.png")


if __name__ == "__main__":
    main()
