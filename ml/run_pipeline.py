"""Run the whole research pipeline in order (Experiments 1-6). Stops at the first failing step.

Usage:  python ml/run_pipeline.py [--csv labels.csv] [--skip-efficientnet]
Each step is also runnable on its own (see README).
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("Dataset validation + patient-level split", "ml/preprocessing/dataset_validation.py"),
    ("Experiment 1: CNN baseline (ResNet50)", "ml/cnn/train_resnet.py"),
    ("(optional) EfficientNet comparison", "ml/cnn/train_efficientnet.py"),
    ("Experiment 2: ViT baseline", "ml/vit/train_vit.py"),
    ("CNN feature extraction", "ml/cnn/extract_cnn_features.py"),
    ("ViT feature extraction", "ml/vit/extract_vit_features.py"),
    ("Experiment 3: CNN + ViT fusion", "ml/fusion/feature_fusion.py"),
    ("Random Forest feature ranking (train split only)", "ml/feature_selection/random_forest_selection.py"),
    ("Experiment 4: XGBoost", "ml/classification/train_xgboost.py"),
    ("Experiment 5: SVM", "ml/classification/train_svm.py"),
    ("Experiment 6: Severity (only with real labels)", "ml/severity/train_severity_xgboost.py"),
    ("Evaluation on the held-out test set", "ml/evaluation/evaluate_models.py"),
    ("SHAP analysis", "ml/explainability/shap_analysis.py"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--skip-efficientnet", action="store_true")
    a = ap.parse_args()
    for title, script in STEPS:
        if a.skip_efficientnet and "efficientnet" in script:
            continue
        cmd = [sys.executable, str(ROOT / script)] + (["--csv", a.csv] if a.csv and "dataset_validation" in script else [])
        print(f"\n=== {title} ===", flush=True)
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            print(f"\nStopped: step failed -> {script}")
            sys.exit(1)
    print("\nPipeline complete. Real results are in results/. Start the app to use the trained models.")


if __name__ == "__main__":
    main()
