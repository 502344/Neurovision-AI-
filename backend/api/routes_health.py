from fastapi import APIRouter

from backend.config import APP_NAME, APP_SUBTITLE, APP_VERSION
from backend.services.model_service import model_status
from ml import config

router = APIRouter()


@router.get("/")
def root():
    return {"name": APP_NAME, "subtitle": APP_SUBTITLE, "version": APP_VERSION,
            "disclaimer": config.MEDICAL_DISCLAIMER,
            "endpoints": ["/health", "/predict", "/validate-image", "/metrics", "/model-info", "/dataset-info", "/docs"]}


@router.get("/health")
def health():
    st = model_status()
    return {"status": "ok", "models_available": st["available"], "missing_artifacts": st["missing"],
            "severity_model_available": st["severity_model_available"], "version": APP_VERSION}
