import base64
import io
import pytest
from PIL import Image
from app import rotate, flip, blur, resize, crop, add_noise


def _make_b64(width=20, height=20):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(100, 150, 200)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _size(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64))).size


def test_rotate_90_swaps_dimensions():
    assert _size(rotate(_make_b64(20, 10), 90)) == (10, 20)


def test_flip_horizontal_returns_valid_image():
    assert isinstance(flip(_make_b64(), "horizontal"), str)


def test_flip_invalid_direction_raises():
    with pytest.raises(ValueError):
        flip(_make_b64(), "diagonal")


def test_blur_returns_nonempty_result():
    assert len(blur(_make_b64(), 2.0)) > 0


def test_resize_changes_dimensions():
    assert _size(resize(_make_b64(20, 20), 40, 10)) == (40, 10)


def test_crop_produces_correct_size():
    assert _size(crop(_make_b64(50, 50), 5, 5, 25, 25)) == (20, 20)


def test_add_noise_preserves_size():
    assert _size(add_noise(_make_b64(30, 30), 0.05)) == (30, 30)
