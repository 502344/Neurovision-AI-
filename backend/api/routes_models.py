import json

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.config import PLOT_WHITELIST
from backend.services.model_service import model_status, read_json
from ml import config

router = APIRouter()


def _plots():
    return {n: f"/plots/{n}" for n in sorted(PLOT_WHITELIST) if (config.RESULTS_DIR / n).exists()}


@router.get("/metrics")
def metrics():
    """Real evaluation results only. If evaluation has not been run, no numbers are returned."""
    path = config.RESULTS_DIR / "model_comparison.csv"
    if not path.exists():
        return {"available": False,
                "message": "MODEL STATUS: Training dataset required. No experimental performance values are displayed "
                           "until real labelled data are supplied and evaluated.",
                "comparison": [], "plots": {}}
    df = pd.read_csv(path)
    rec = read_json(config.RESULTS_DIR / "evaluation_record.json") or {}
    focus = rec.get("focus_model")
    row = df[df["Model"] == focus].iloc[0].to_dict() if focus and (df["Model"] == focus).any() else None
    m = pd.read_csv(config.RESULTS_DIR / "metrics.csv") if (config.RESULTS_DIR / "metrics.csv").exists() else None
    cm = None
    if m is not None and focus and (m.model == focus).any():
        r = m[m.model == focus].iloc[0]
        cm = {"tn": int(r.tn), "fp": int(r.fp), "fn": int(r.fn), "tp": int(r.tp)}
    return {"available": True, "focus_model": focus, "focus_metrics": row,
            "comparison": json.loads(df.to_json(orient="records")), "confusion_matrix": cm,
            "evaluation": rec, "severity_test_metrics": read_json(config.RESULTS_DIR / "severity_test_metrics.json"),
            "plots": _plots()}


@router.get("/model-info")
def model_info():
    st = model_status()
    meta = read_json(config.PIPELINE_META_PATH)
    return {**st, "pipeline": None if not meta else {k: meta[k] for k in (
        "created_utc", "image_size", "fusion_method", "cnn_dim", "vit_dim", "fused_dim", "selected_k",
        "decision_threshold", "train_size", "validation_size", "test_size", "random_state", "versions",
        "vit_pretrained_preset") if k in meta},
        "architecture": "Retinal Fundus Image → Preprocessing → CNN + ViT Feature Extraction → Feature Fusion → "
                        "Random Forest Feature Ranking → XGBoost Classification → ASD/Non-ASD → Severity Estimation "
                        "(where labelled data are available) → Grad-CAM → Automated Report",
        "severity_status": read_json(config.RESULTS_DIR / "severity_status.json"),
        "feature_selection": read_json(config.RESULTS_DIR / "feature_selection_record.json"),
        "disclaimer": config.MEDICAL_DISCLAIMER}


@router.get("/dataset-info")
def dataset_info():
    rep = read_json(config.RESULTS_DIR / "dataset_report.json")
    if not rep:
        return {"available": False, "message": "Dataset has not been validated yet. Run "
                "'python ml/preprocessing/dataset_validation.py' with a real labelled dataset."}
    return {"available": True, **rep}


@router.get("/plots/{name}")
def plot(name: str):
    if name not in PLOT_WHITELIST or not (config.RESULTS_DIR / name).exists():
        raise HTTPException(404, "Plot not available. Run the evaluation pipeline first.")
    return FileResponse(config.RESULTS_DIR / name, media_type="image/png")
