"""Backend settings (re-exports ML paths so the API and the training code cannot drift apart)."""
import os

from ml import config as ml_config

CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "NEUROVISION_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
KEEP_UPLOADS = os.environ.get("NEUROVISION_KEEP_UPLOADS", "0") == "1"   # default: delete raw uploads after analysis
APP_NAME = "NEUROVISION AI"
APP_SUBTITLE = "AI-Based Retinal Image Analysis Framework for Early Screening of Autism Spectrum Disorder"
APP_VERSION = "1.0.0"
PLOT_WHITELIST = {"confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png", "training_history.png",
                  "feature_importance.png", "topk_sweep.png", "shap_global_importance.png", "shap_beeswarm.png"}
__all__ = ["ml_config", "CORS_ORIGINS", "KEEP_UPLOADS", "APP_NAME", "APP_SUBTITLE", "APP_VERSION", "PLOT_WHITELIST"]
