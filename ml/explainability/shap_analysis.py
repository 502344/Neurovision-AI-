"""SHAP analysis of the XGBoost classifier (global + per-feature contribution direction).

SHAP values describe the model's mathematical dependence on abstract deep-feature dimensions. They are NOT
medical explanations and must not be translated into clinical claims.
Run:  python ml/explainability/shap_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml import config  # noqa: E402


def main():
    config.ensure_dirs()
    try:
        import shap
        from xgboost import XGBClassifier
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[shap] Missing dependency: {exc}. pip install shap")
        sys.exit(1)
    try:
        X = np.load(config.FEATURES_DIR / "fused_features.npy")
        sel = np.load(config.SELECTED_IDX_PATH)
        meta = pd.read_csv(config.FEATURES_DIR / "feature_metadata.csv")
    except FileNotFoundError as exc:
        print(f"[shap] Missing file {exc.filename}.")
        sys.exit(1)
    if not config.XGB_MODEL_PATH.exists():
        print("[shap] xgboost_asd.json missing. Train XGBoost first.")
        sys.exit(1)
    clf = XGBClassifier()
    clf.load_model(str(config.XGB_MODEL_PATH))
    rows = (meta.split == "test").values if (meta.split == "test").any() else (meta.split == "validation").values
    Xs = X[rows][:, sel]
    names = [f"f{int(i)}" for i in sel]
    sv = shap.TreeExplainer(clf).shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else sv
    pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0),
                  "mean_signed_shap": sv.mean(0)}).sort_values("mean_abs_shap", ascending=False)\
        .to_csv(config.RESULTS_DIR / "shap_importance.csv", index=False)
    plt.figure(); shap.summary_plot(sv, Xs, feature_names=names, plot_type="bar", show=False, max_display=20)
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "shap_global_importance.png", dpi=150); plt.close()
    plt.figure(); shap.summary_plot(sv, Xs, feature_names=names, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "shap_beeswarm.png", dpi=150); plt.close()
    print(f"[shap] saved SHAP plots for {len(Xs)} samples ({'test' if (meta.split=='test').any() else 'validation'} split)")


if __name__ == "__main__":
    main()
