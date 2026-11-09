"""Simple, transparent image-quality heuristics (not a clinical gradeability score)."""
from __future__ import annotations

import cv2
import numpy as np


def assess_quality(image: np.ndarray, *, blur_threshold: float = 40.0,
                   dark_threshold: float = 25.0, bright_threshold: float = 235.0) -> dict:
    """Return sharpness, brightness, contrast and human-readable warnings for an RGB uint8 image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    fundus = gray[gray > 12]
    brightness = float(fundus.mean()) if fundus.size else float(gray.mean())
    contrast = float(fundus.std()) if fundus.size else float(gray.std())
    warnings = []
    if lap_var < blur_threshold:
        warnings.append("Image appears blurred (low sharpness).")
    if brightness < dark_threshold:
        warnings.append("Image appears very dark.")
    if brightness > bright_threshold:
        warnings.append("Image appears over-exposed.")
    if contrast < 10:
        warnings.append("Image has very low contrast.")
    return {"sharpness": round(lap_var, 2), "brightness": round(brightness, 2),
            "contrast": round(contrast, 2), "warnings": warnings, "acceptable": not warnings}
