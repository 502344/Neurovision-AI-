import numpy as np

from ml.preprocessing.image_preprocessing import (crop_retinal_region, load_image, normalize_image,
                                                  preprocess_image, resize_image)
from ml.preprocessing.image_quality import assess_quality


def test_preprocessing(sample_image):
    out = preprocess_image(sample_image, size=64)
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8


def test_default_size_is_224(sample_image):
    assert preprocess_image(sample_image).shape == (224, 224, 3)


def test_black_border_crop(sample_image):
    img = load_image(sample_image)
    cropped = crop_retinal_region(img)
    assert cropped.shape[0] < img.shape[0] and cropped.shape[1] < img.shape[1]


def test_normalize_and_resize():
    a = np.full((10, 20, 3), 255, np.uint8)
    assert normalize_image(a).max() == 1.0
    assert resize_image(a, 32).shape == (32, 32, 3)


def test_quality_reports_dark_image():
    q = assess_quality(np.zeros((64, 64, 3), np.uint8))
    assert not q["acceptable"] and q["warnings"]
