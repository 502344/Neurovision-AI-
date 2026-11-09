"""Train XGBoost on selected CNN+ViT features (plus an ablation on the full fused vector).

Hyper-parameters are chosen on the VALIDATION split; the test split is untouched.
Run:  python ml/classification/train_xgboost.py
Saves models/xgboost_asd.json, models/xgboost_fused_full.json, models/pipeline_meta.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml import config  # noqa: E402
from ml.evaluation.metrics import binary_metrics  # noqa: E402


def fit_xgb(Xtr, ytr, Xva, yva, seed=None):
    """Small validation-driven grid over depth; early stopping on validation logloss."""
    from xgboost import XGBClassifier
    seed = config.RANDOM_STATE if seed is None else seed
    pos, neg = int((ytr == 1).sum()), int((ytr == 0).sum())
    spw = neg / max(pos, 1)
    has_val = Xva is not None and len(yva) > 0 and len(np.unique(yva)) > 1
    best, best_score, grid_log = None, -np.inf, []
    for depth in (2, 3, 4):
        clf = XGBClassifier(n_estimators=500, max_depth=depth, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=2, reg_lambda=2.0, scale_pos_weight=spw,
                            eval_metric="logloss", random_state=seed, n_jobs=1,
                            early_stopping_rounds=30 if has_val else None)
        if has_val:
            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            score = binary_metrics(yva, clf.predict_proba(Xva)[:, 1])["roc_auc"]
        else:
            clf.fit(Xtr, ytr, verbose=False)
            score = 0.0
        grid_log.append({"max_depth": depth, "val_roc_auc": score,
                         "best_iteration": int(getattr(clf, "best_iteration", clf.n_estimators) or 0)})
        if score > best_score:
            best, best_score = clf, score
    return best, grid_log, has_val


def main():
    config.ensure_dirs()
    config.set_seeds()
    try:
        X = np.load(config.FEATURES_DIR / "fused_features.npy")
        sel = np.load(config.SELECTED_IDX_PATH)
        meta = pd.read_csv(config.FEATURES_DIR / "feature_metadata.csv")
        n_cnn = np.load(config.FEATURES_DIR / "cnn_features.npy", mmap_mode="r").shape[1]
        n_vit = np.load(config.FEATURES_DIR / "vit_features.npy", mmap_mode="r").shape[1]
    except FileNotFoundError as exc:
        print(f"[xgboost] Missing file {exc.filename}. Run extraction, fusion and RF selection first.")
        sys.exit(1)
    tr, va = (meta.split == "train").values, (meta.split == "validation").values
    y = meta["label_idx"].values
    if len(np.unique(y[tr])) < 2:
        print("[xgboost] Training split has a single class.")
        sys.exit(1)

    records = {}
    for tag, cols, path in (("selected", sel, config.XGB_MODEL_PATH),
                            ("fused_full", np.arange(X.shape[1]), config.XGB_FULL_MODEL_PATH)):
        clf, grid, has_val = fit_xgb(X[tr][:, cols], y[tr], X[va][:, cols], y[va])
        clf.save_model(str(path))
        vm = binary_metrics(y[va], clf.predict_proba(X[va][:, cols])[:, 1]) if has_val else None
        records[tag] = {"n_features": int(len(cols)), "grid": grid, "validation_metrics": vm,
                        "params": {k: v for k, v in clf.get_params().items() if k in
                                   ("max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree")}}
        print(f"[xgboost:{tag}] saved {path.name}; val AUC={vm['roc_auc'] if vm else 'n/a'}")

    import xgboost, sklearn, tensorflow
    pipeline_meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image_size": config.IMAGE_SIZE, "fusion_method": config.FUSION_METHOD,
        "fusion_cnn_weight": config.FUSION_CNN_WEIGHT,
        "cnn_dim": int(n_cnn), "vit_dim": int(n_vit), "fused_dim": int(X.shape[1]),
        "selected_k": int(len(sel)), "decision_threshold": config.DECISION_THRESHOLD,
        "class_names": config.CLASS_NAMES, "cnn_backbone": "resnet50",
        "crop_black_border": config.CROP_BLACK_BORDER,
        "vit_pretrained_preset": config.VIT_PRESET if config.USE_PRETRAINED else "random-init (NOT for research)",
        "train_size": int(tr.sum()), "validation_size": int(va.sum()), "test_size": int((meta.split == "test").sum()),
        "random_state": config.RANDOM_STATE,
        "versions": {"xgboost": xgboost.__version__, "sklearn": sklearn.__version__, "tensorflow": tensorflow.__version__},
        "training_records": records, "config": config.as_dict(),
    }
    config.PIPELINE_META_PATH.write_text(json.dumps(pipeline_meta, indent=2, default=str))
    print(f"[xgboost] pipeline metadata -> {config.PIPELINE_META_PATH}")
if __name__ == "__main__":
    main()
