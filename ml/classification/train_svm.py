"""Train an RBF-SVM on the same selected features for a like-for-like comparison with XGBoost.

C is chosen on the validation split. Run:  python ml/classification/train_svm.py
Saves models/svm_asd.joblib
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml import config  # noqa: E402
from ml.evaluation.metrics import binary_metrics  # noqa: E402


def main():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    config.ensure_dirs()
    config.set_seeds()
    try:
        X = np.load(config.FEATURES_DIR / "fused_features.npy")
        sel = np.load(config.SELECTED_IDX_PATH)
        meta = pd.read_csv(config.FEATURES_DIR / "feature_metadata.csv")
    except FileNotFoundError as exc:
        print(f"[svm] Missing file {exc.filename}. Run extraction, fusion and RF selection first.")
        sys.exit(1)
    tr, va = (meta.split == "train").values, (meta.split == "validation").values
    y = meta["label_idx"].values
    if len(np.unique(y[tr])) < 2:
        print("[svm] Training split has a single class.")
        sys.exit(1)
    Xs = X[:, sel]
    has_val = va.sum() > 0 and len(np.unique(y[va])) > 1
    best, best_score, log = None, -np.inf, []
    for C in (0.1, 1.0, 10.0):
        pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma="scale", probability=True,
                                                   class_weight="balanced", random_state=config.RANDOM_STATE))
        pipe.fit(Xs[tr], y[tr])
        score = binary_metrics(y[va], pipe.predict_proba(Xs[va])[:, 1])["roc_auc"] if has_val else 0.0
        log.append({"C": C, "val_roc_auc": score})
        if score > best_score:
            best, best_score = pipe, score
    joblib.dump(best, config.SVM_MODEL_PATH)
    (config.RESULTS_DIR / "svm_training_record.json").write_text(json.dumps(
        {"grid": log, "n_features": int(len(sel)), "selection": "validation ROC-AUC"}, indent=2))
    print(f"[svm] saved {config.SVM_MODEL_PATH.name}; grid={log}")


if __name__ == "__main__":
    main()
