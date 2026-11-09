"""Severity model (Mild / Moderate / Severe) -- trained ONLY when real severity labels exist.

Severity is never generated, guessed, or derived from prediction confidence. If labelled data are
insufficient this script exits gracefully and the application reports "Severity: Not Available".

Run:  python ml/severity/train_severity_xgboost.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml import config  # noqa: E402
from ml.evaluation.metrics import multiclass_metrics  # noqa: E402

NOT_AVAILABLE = ("Severity: Not Available\n\nReason:\nThe dataset does not contain validated severity labels "
                 "(or too few per class: {detail}).")


def severity_data_status(meta: pd.DataFrame) -> tuple[bool, str]:
    """Return (usable, explanation) based on real labels in the TRAIN split."""
    tr = meta[(meta.split == "train") & (meta.label == "ASD") & meta.severity.isin(config.SEVERITY_CLASSES)]
    counts = tr.severity.value_counts().to_dict()
    if not counts:
        return False, "no severity labels"
    missing = [c for c in config.SEVERITY_CLASSES if counts.get(c, 0) < config.MIN_SEVERITY_SAMPLES_PER_CLASS]
    if missing:
        return False, f"classes with fewer than {config.MIN_SEVERITY_SAMPLES_PER_CLASS} training samples: {missing}, counts={counts}"
    return True, f"counts={counts}"


def main():
    config.ensure_dirs()
    config.set_seeds()
    if config.SEVERITY_MODEL_PATH.exists():
        config.SEVERITY_MODEL_PATH.unlink()            # never leave a stale model from another dataset
    try:
        meta = pd.read_csv(config.FEATURES_DIR / "feature_metadata.csv")
        X = np.load(config.FEATURES_DIR / "fused_features.npy")
        sel = np.load(config.SELECTED_IDX_PATH)
    except FileNotFoundError as exc:
        print(f"[severity] Missing file {exc.filename}. Run extraction, fusion and RF selection first.")
        sys.exit(1)
    ok, detail = severity_data_status(meta)
    if not ok:
        print(NOT_AVAILABLE.format(detail=detail))
        (config.RESULTS_DIR / "severity_status.json").write_text(json.dumps({"available": False, "reason": detail}))
        sys.exit(0)

    from xgboost import XGBClassifier
    lab = {c: i for i, c in enumerate(config.SEVERITY_CLASSES)}
    has = meta.severity.isin(config.SEVERITY_CLASSES) & (meta.label == "ASD")
    tr, va = (meta.split == "train") & has, (meta.split == "validation") & has
    Xs, y = X[:, sel], meta.severity.map(lab).values
    clf = XGBClassifier(objective="multi:softprob", num_class=3, n_estimators=300, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                        random_state=config.RANDOM_STATE, n_jobs=1,
                        early_stopping_rounds=30 if va.sum() > 0 else None)
    # inverse-frequency sample weights computed from the TRAIN split only
    cnt = np.bincount(y[tr].astype(int), minlength=3).astype(float)
    sw = np.array([len(y[tr]) / (3 * cnt[int(c)]) for c in y[tr]])
    if va.sum() > 0:
        clf.fit(Xs[tr], y[tr].astype(int), sample_weight=sw, eval_set=[(Xs[va], y[va].astype(int))], verbose=False)
    else:
        clf.fit(Xs[tr], y[tr].astype(int), sample_weight=sw, verbose=False)
    clf.save_model(str(config.SEVERITY_MODEL_PATH))
    rec = {"available": True, "classes": config.SEVERITY_CLASSES, "train_counts": detail,
           "n_validation": int(va.sum()), "note": "Severity labels come from the supplied dataset; not validated by us."}
    if va.sum() > 0:
        rec["validation_metrics"] = multiclass_metrics(y[va].astype(int), clf.predict(Xs[va]), [0, 1, 2])
    (config.RESULTS_DIR / "severity_status.json").write_text(json.dumps(rec, indent=2))
    print(f"[severity] trained on real labels ({detail}); saved {config.SEVERITY_MODEL_PATH.name}")


if __name__ == "__main__":
    main()
