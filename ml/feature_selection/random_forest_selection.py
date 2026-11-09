"""Random Forest feature ranking on fused CNN+ViT features.

Fused features -> Random Forest (TRAIN split only) -> importance -> rank -> top-K.
K is chosen by validation ROC-AUC over TOP_K_CANDIDATES when AUTO_TOPK is on, else TOP_K_FEATURES is used.
The test split is never used here.

Run:  python ml/feature_selection/random_forest_selection.py
Saves results/feature_importance.csv, features/selected_features.npy, results/topk_sweep.csv/.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml import config  # noqa: E402
from ml.evaluation.metrics import binary_metrics  # noqa: E402
from ml.evaluation.plots import plot_feature_importance, plot_topk_sweep  # noqa: E402


def rank_features(X_train, y_train, n_estimators=None, seed=None) -> np.ndarray:
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=n_estimators or config.RF_N_ESTIMATORS, class_weight="balanced",
                                n_jobs=-1, random_state=config.RANDOM_STATE if seed is None else seed)
    rf.fit(X_train, y_train)
    return rf.feature_importances_


def quick_val_auc(Xtr, ytr, Xva, yva) -> float | None:
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", random_state=config.RANDOM_STATE, n_jobs=1)
    clf.fit(Xtr, ytr)
    return binary_metrics(yva, clf.predict_proba(Xva)[:, 1])["roc_auc"]


def main():
    config.ensure_dirs()
    config.set_seeds()
    try:
        X = np.load(config.FEATURES_DIR / "fused_features.npy")
        meta = pd.read_csv(config.FEATURES_DIR / "feature_metadata.csv")
    except FileNotFoundError as exc:
        print(f"[rf] Missing file {exc.filename}. Run extraction + fusion first.")
        sys.exit(1)
    tr, va = (meta.split == "train").values, (meta.split == "validation").values
    y = meta["label_idx"].values
    if len(np.unique(y[tr])) < 2:
        print("[rf] Training split has a single class; cannot rank features.")
        sys.exit(1)

    imp = rank_features(X[tr], y[tr])
    order = np.argsort(imp)[::-1]
    n_cnn = np.load(config.FEATURES_DIR / "cnn_features.npy", mmap_mode="r").shape[1]
    pd.DataFrame({"rank": np.arange(1, len(order) + 1), "feature_index": order, "importance": imp[order],
                  "source": np.where(order < n_cnn, "CNN", "ViT")}).to_csv(config.FEATURE_IMPORTANCE_CSV, index=False)
    plot_feature_importance(imp, config.RESULTS_DIR / "feature_importance.png")

    candidates = sorted({min(k, X.shape[1]) for k in config.TOP_K_CANDIDATES + [config.TOP_K_FEATURES]})
    sweep = []
    can_validate = va.sum() > 0 and len(np.unique(y[va])) > 1
    if can_validate:
        for k in candidates:
            sel = order[:k]
            sweep.append({"k": int(k), "val_roc_auc": quick_val_auc(X[tr][:, sel], y[tr], X[va][:, sel], y[va])})
        pd.DataFrame(sweep).to_csv(config.RESULTS_DIR / "topk_sweep.csv", index=False)
        plot_topk_sweep(sweep, config.RESULTS_DIR / "topk_sweep.png")
    if config.AUTO_TOPK and can_validate:
        best = max(sweep, key=lambda r: (r["val_roc_auc"], -r["k"]))
        k = best["k"]
        how = "selected by validation ROC-AUC"
    else:
        k = min(config.TOP_K_FEATURES, X.shape[1])
        how = "fixed by TOP_K_FEATURES" + ("" if can_validate else " (validation split unusable for a sweep)")
    selected = order[:k].astype(np.int64)
    np.save(config.SELECTED_IDX_PATH, selected)
    (config.RESULTS_DIR / "feature_selection_record.json").write_text(json.dumps({
        "fused_dim": int(X.shape[1]), "selected_k": int(k), "how": how, "sweep": sweep,
        "cnn_features_selected": int((selected < n_cnn).sum()),
        "vit_features_selected": int((selected >= n_cnn).sum()),
        "fit_on": "train split only", "random_state": config.RANDOM_STATE}, indent=2))
    print(f"[rf] fused_dim={X.shape[1]} selected K={k} ({how}); sweep={sweep}")


if __name__ == "__main__":
    main()
