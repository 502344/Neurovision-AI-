"""Image validation and preprocessing for retinal fundus images.

Output convention: ``preprocess_image`` returns an ``H x W x 3`` uint8 RGB array
(default 224 x 224). Backbone-specific scaling (ResNet caffe-style, ViT [-1, 1])
is performed *inside* each Keras model so training and inference cannot diverge.
``normalize_image`` (to [0, 1]) is provided for utilities/visualisation.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from ml import config

Image.MAX_IMAGE_PIXELS = 200_000_000


class InvalidImageError(ValueError):
    """Raised when an image cannot be used for analysis."""


def validate_image(path, *, min_side: int | None = None, max_mb: float | None = None) -> dict:
    """Check extension, size and integrity. Returns metadata or raises InvalidImageError."""
    p = Path(path)
    min_side = config.MIN_IMAGE_SIDE if min_side is None else min_side
    max_mb = config.MAX_FILE_SIZE_MB if max_mb is None else max_mb
    if not p.exists() or not p.is_file():
        raise InvalidImageError(f"File not found: {p.name}")
    if p.suffix.lower() not in config.ALLOWED_EXTENSIONS:
        raise InvalidImageError(
            f"Unsupported file type '{p.suffix}'. Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}")
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb == 0:
        raise InvalidImageError("File is empty.")
    if size_mb > max_mb:
        raise InvalidImageError(f"File too large ({size_mb:.1f} MB). Limit is {max_mb} MB.")
    try:
        with Image.open(p) as im:
            im.verify()                     # structural integrity
        with Image.open(p) as im:
            im.load()                       # full decode (catches truncated files)
            width, height = im.size
            mode = im.mode
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError(f"Image is corrupted or unreadable: {exc}") from exc
    if min(width, height) < min_side:
        raise InvalidImageError(f"Image too small ({width}x{height}); minimum side is {min_side}px.")
    return {"path": str(p), "width": width, "height": height, "mode": mode, "size_mb": round(size_mb, 3)}


def load_image(path) -> np.ndarray:
    """Load any supported image as an RGB uint8 array (handles gray/RGBA/16-bit)."""
    try:
        with Image.open(path) as im:
            im.load()
            if im.mode in ("I;16", "I;16B", "I;16L", "I"):
                arr = np.asarray(im).astype(np.float32)
                arr = 255.0 * (arr - arr.min()) / max(float(arr.max() - arr.min()), 1.0)
                im = Image.fromarray(arr.astype(np.uint8))
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"Cannot load image: {exc}") from exc


def crop_retinal_region(image: np.ndarray, threshold: int | None = None) -> np.ndarray:
    """Remove the black border surrounding the fundus disc (bounding-box crop)."""
    threshold = config.BLACK_THRESHOLD if threshold is None else threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    if mask.sum() < 0.05 * mask.size:      # nearly all dark: do not crop blindly
        return image
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if (y1 - y0) < 32 or (x1 - x0) < 32:
        return image
    return image[y0:y1, x0:x1]


def resize_image(image: np.ndarray, size: int | None = None) -> np.ndarray:
    size = config.IMAGE_SIZE if size is None else size
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> float32 [0,1]."""
    return image.astype(np.float32) / 255.0


def preprocess_image(path, *, size: int | None = None, crop: bool | None = None) -> np.ndarray:
    """validate -> load -> (crop) -> resize. Returns uint8 RGB (size, size, 3)."""
    validate_image(path)
    img = load_image(path)
    if config.CROP_BLACK_BORDER if crop is None else crop:
        img = crop_retinal_region(img)
    return resize_image(img, size)
