"""Evaluate all experiments on the SAME held-out test split with real predictions.

Experiments
  1 CNN baseline  2 ViT baseline  3 CNN+ViT (fused, no selection)  4 CNN+ViT+RF+XGBoost
  5 CNN+ViT+RF+SVM  6 Severity (only when labelled data existed at training time)

Run:  python ml/evaluation/evaluate_models.py
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
from ml.data import make_dataset  # noqa: E402
from ml.evaluation.metrics import binary_metrics, multiclass_metrics  # noqa: E402
from ml.evaluation.plots import plot_confusion_matrix, plot_pr_curves, plot_roc_curves  # noqa: E402
from ml.models import load_keras_model  # noqa: E402

MAIN = "CNN + ViT + RF + XGBoost"


def _keras_probs(model_path, paths):
    model = load_keras_model(model_path)
    return model.predict(make_dataset(paths, None, cache=False), verbose=0).ravel()


def main():
    from xgboost import XGBClassifier
    config.ensure_dirs()
    config.set_seeds()
    try:
        meta = pd.read_csv(config.FEATURES_DIR / "feature_metadata.csv")
        fused = np.load(config.FEATURES_DIR / "fused_features.npy")
        sel = np.load(config.SELECTED_IDX_PATH)
    except FileNotFoundError as exc:
        print(f"[evaluate] Missing file {exc.filename}. Complete the training pipeline first.")
        sys.exit(1)
    test = (meta.split == "test").values
    if test.sum() == 0 or len(np.unique(meta.label_idx[test])) < 2:
        print("[evaluate] The test split is empty or single-class; metrics are undefined. No results written.")
        sys.exit(1)
    y = meta.label_idx.values[test]
    test_paths = meta.image_path.values[test]
    preds: dict[str, np.ndarray] = {}

    if config.CNN_MODEL_PATH.exists():
        preds["CNN (ResNet50)"] = _keras_probs(config.CNN_MODEL_PATH, test_paths)
    if config.EFFNET_MODEL_PATH.exists():
        preds["CNN (EfficientNetB0)"] = _keras_probs(config.EFFNET_MODEL_PATH, test_paths)
    if config.VIT_MODEL_PATH.exists():
        preds["ViT"] = _keras_probs(config.VIT_MODEL_PATH, test_paths)
    if config.XGB_FULL_MODEL_PATH.exists():
        m = XGBClassifier(); m.load_model(str(config.XGB_FULL_MODEL_PATH))
        preds["CNN + ViT"] = m.predict_proba(fused[test])[:, 1]
    if config.XGB_MODEL_PATH.exists():
        m = XGBClassifier(); m.load_model(str(config.XGB_MODEL_PATH))
        preds[MAIN] = m.predict_proba(fused[test][:, sel])[:, 1]
    if config.SVM_MODEL_PATH.exists():
        preds["CNN + ViT + RF + SVM"] = joblib.load(config.SVM_MODEL_PATH).predict_proba(fused[test][:, sel])[:, 1]
    if not preds:
        print("[evaluate] No trained models found."); sys.exit(1)

    rows = []
    for name, p in preds.items():
        rows.append({"model": name, **binary_metrics(y, p, config.DECISION_THRESHOLD)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(config.RESULTS_DIR / "metrics.csv", index=False)
    comp = metrics.rename(columns={"model": "Model", "accuracy": "Accuracy", "precision": "Precision",
                                   "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC",
                                   "sensitivity": "Sensitivity", "specificity": "Specificity"})
    comp = comp[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Sensitivity", "Specificity", "n"]]
    comp.to_csv(config.RESULTS_DIR / "model_comparison.csv", index=False)
    pd.DataFrame({"image_path": test_paths, "true_label_idx": y,
                  **{f"prob_{k}": v for k, v in preds.items()}}).to_csv(config.RESULTS_DIR / "predictions_test.csv", index=False)

    curves = {k: (y, v) for k, v in preds.items()}
    plot_roc_curves(curves, config.RESULTS_DIR / "roc_curve.png")
    plot_pr_curves(curves, config.RESULTS_DIR / "precision_recall_curve.png")
    focus = MAIN if MAIN in preds else next(iter(preds))
    r = metrics[metrics.model == focus].iloc[0]
    plot_confusion_matrix([[r.tn, r.fp], [r.fn, r.tp]], config.RESULTS_DIR / "confusion_matrix.png",
                          title=f"{focus}\n(test set, n={int(r.n)})")

    record = {"test_n": int(test.sum()), "test_ASD": int((y == 1).sum()), "test_Non_ASD": int((y == 0).sum()),
              "test_set_rebalanced": False, "same_test_set_for_all_models": True, "focus_model": focus,
              "decision_threshold": config.DECISION_THRESHOLD, "dataset_report": None, "severity": None,
              "caveats": ["Single hold-out split; no external validation.",
                          "Small test sets yield wide confidence intervals; differences between models may be noise."]}
    rp = config.RESULTS_DIR / "dataset_report.json"
    if rp.exists():
        d = json.loads(rp.read_text()); record["dataset_report"] = {
            k: d.get(k) for k in ("split_strategy", "patient_ids_available", "warnings")}
    # bootstrap 95% CI for ROC-AUC of every model (test set only, fixed seed)
    rng = np.random.default_rng(config.RANDOM_STATE)
    ci = {}
    for name, p in preds.items():
        aucs = []
        for _ in range(500):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) == 2:
                aucs.append(binary_metrics(y[idx], p[idx])["roc_auc"])
        ci[name] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))] if aucs else None
    record["roc_auc_bootstrap_95ci"] = ci

    if config.SEVERITY_MODEL_PATH.exists():
        sev = (meta.split == "test") & (meta.label == "ASD") & meta.severity.isin(config.SEVERITY_CLASSES)
        if sev.sum() > 0:
            m = XGBClassifier(); m.load_model(str(config.SEVERITY_MODEL_PATH))
            ys = meta.severity[sev].map({c: i for i, c in enumerate(config.SEVERITY_CLASSES)}).values
            record["severity"] = multiclass_metrics(ys, m.predict(fused[sev.values][:, sel]), [0, 1, 2])
            (config.RESULTS_DIR / "severity_test_metrics.json").write_text(json.dumps(record["severity"], indent=2))
    (config.RESULTS_DIR / "evaluation_record.json").write_text(json.dumps(record, indent=2))
    print(comp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n[evaluate] test n={record['test_n']} (ASD={record['test_ASD']}, Non-ASD={record['test_Non_ASD']}). "
          "Results written to results/.")


if __name__ == "__main__":
    main()
