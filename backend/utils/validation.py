"""Upload validation: extension, MIME type, size, integrity."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

from ml import config
from ml.preprocessing.image_preprocessing import InvalidImageError, validate_image


async def save_and_validate_upload(file: UploadFile) -> tuple[Path, dict]:
    """Stream the upload to uploads/ (enforcing the size limit) and validate it. Raises HTTPException."""
    name = file.filename or ""
    ext = Path(name).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file extension '{ext or 'none'}'. "
                                 f"Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}")
    ctype = (file.content_type or "").lower()
    if ctype not in config.ALLOWED_MIME_TYPES:
        raise HTTPException(415, f"Unsupported content type '{ctype or 'unknown'}'. "
                                 f"Allowed: {', '.join(sorted(config.ALLOWED_MIME_TYPES))}")
    config.ensure_dirs()
    dest = config.UPLOADS_DIR / f"{uuid.uuid4().hex}{ext}"       # original filename is never stored
    limit = config.MAX_FILE_SIZE_MB * 1024 * 1024
    written = 0
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(1 << 20):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"File too large. Limit is {config.MAX_FILE_SIZE_MB} MB.")
                out.write(chunk)
        if written == 0:
            raise HTTPException(400, "The uploaded file is empty.")
        info = validate_image(dest)
    except InvalidImageError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    return dest, info
