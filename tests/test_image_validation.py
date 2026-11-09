import pytest

from ml.preprocessing.image_preprocessing import InvalidImageError, validate_image


def test_image_validation(sample_image):
    info = validate_image(sample_image)
    assert info["width"] == 96 and info["height"] == 96


def test_invalid_image_corrupted(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"this is not an image")
    with pytest.raises(InvalidImageError):
        validate_image(bad)


def test_invalid_image_extension(tmp_path):
    p = tmp_path / "x.gif"
    p.write_bytes(b"GIF89a")
    with pytest.raises(InvalidImageError, match="Unsupported"):
        validate_image(p)


def test_invalid_image_missing_and_empty(tmp_path):
    with pytest.raises(InvalidImageError):
        validate_image(tmp_path / "nope.png")
    e = tmp_path / "e.png"
    e.write_bytes(b"")
    with pytest.raises(InvalidImageError, match="empty"):
        validate_image(e)


def test_invalid_image_too_small(tmp_path):
    from PIL import Image
    p = tmp_path / "s.png"
    Image.new("RGB", (16, 16)).save(p)
    with pytest.raises(InvalidImageError, match="too small"):
        validate_image(p)
