"""NEUROVISION AI backend.   Run:  uvicorn backend.main:app --reload"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api import routes_health, routes_models, routes_prediction
from backend.config import APP_NAME, APP_SUBTITLE, APP_VERSION, CORS_ORIGINS
from backend.utils.logging_utils import setup_logging
from ml import config

config.ensure_dirs()
log = setup_logging()

app = FastAPI(title=APP_NAME, description=APP_SUBTITLE + "\n\n" + config.MEDICAL_DISCLAIMER, version=APP_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])
app.mount("/files/gradcam", StaticFiles(directory=config.GRADCAM_DIR), name="gradcam")
app.mount("/files/reports", StaticFiles(directory=config.REPORTS_DIR), name="reports")
app.include_router(routes_health.router)
app.include_router(routes_prediction.router)
app.include_router(routes_models.router)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
