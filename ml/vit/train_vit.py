"""Train the ViT baseline: ViT backbone -> [CLS] token -> Dense -> ASD / Non-ASD.

The backbone is frozen first; optional fine-tuning follows (FINE_TUNE_EPOCHS > 0 and pretrained weights).
Run:  python ml/vit/train_vit.py
Saves models/vit_best.keras
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml import config  # noqa: E402
from ml.models import build_vit_classifier  # noqa: E402
from ml.training_utils import train_classifier  # noqa: E402


def main():
    config.set_seeds()
    model = build_vit_classifier()
    train_classifier(model, name="vit", best_path=config.VIT_MODEL_PATH, frozen_layer_names=[])


if __name__ == "__main__":
    main()
