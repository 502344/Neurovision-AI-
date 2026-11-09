import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.services.model_service import ModelNotAvailableError
from backend.services.prediction_service import predict_retinal_image
from backend.utils.validation import save_and_validate_upload
from backend.config import KEEP_UPLOADS
from ml.preprocessing.image_preprocessing import InvalidImageError, load_image, crop_retinal_region
from ml.preprocessing.image_quality import assess_quality
from ml import config

router = APIRouter()
log = logging.getLogger("neurovision.predict")


@router.post("/validate-image")
async def validate_image_endpoint(file: UploadFile = File(...)):
    """Validate integrity/format/size and report quality hints without running any model."""
    path, info = await save_and_validate_upload(file)
    try:
        quality = assess_quality(crop_retinal_region(load_image(path)) if config.CROP_BLACK_BORDER else load_image(path))
        return {"valid": True, "width": info["width"], "height": info["height"], "size_mb": info["size_mb"],
                "quality": quality}
    except InvalidImageError as exc:
        raise HTTPException(422, str(exc))
    finally:
        path.unlink(missing_ok=True)


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """AI-assisted screening prediction for one retinal fundus image (not a diagnosis)."""
    path, _ = await save_and_validate_upload(file)
    try:
        return await _run(path)
    finally:
        if not KEEP_UPLOADS:
            path.unlink(missing_ok=True)


async def _run(path):
    from starlette.concurrency import run_in_threadpool
    try:
        return await run_in_threadpool(predict_retinal_image, path)
    except ModelNotAvailableError as exc:
        raise HTTPException(503, str(exc))
    except InvalidImageError as exc:
        raise HTTPException(422, str(exc))
    except Exception:                                    # never leak internals; log for the developer
        log.exception("Prediction failed")
        raise HTTPException(500, "Prediction failed due to an internal error. See backend log.")
