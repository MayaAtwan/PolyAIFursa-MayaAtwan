# Tests for POST /predict
#
# What we check:
#   - Happy path: a valid JPEG upload triggers detection and returns structured data.
#   - Validation: non-image files (e.g. PDF) are rejected with 400 before any
#     file I/O or model inference happens.
#
# How we test it:
#   - The real YOLO model is replaced with FakeModel (from conftest) via
#     monkeypatch so tests run offline without GPU.
#   - make_jpeg_bytes() builds a minimal in-memory JPEG — no fixture files needed.

import app as app_module
from tests.conftest import FakeModel, make_jpeg_bytes


def test_predict_returns_prediction_data(client, monkeypatch):
    monkeypatch.setattr(app_module, "model", FakeModel())

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", make_jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200

    data = response.json()
    assert "prediction_uid" in data
    assert data["detection_count"] == 1
    assert data["labels"] == ["person"]


def test_predict_rejects_non_image_file(client):
    response = client.post(
        "/predict",
        files={"file": ("document.pdf", b"fake pdf content", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Only image files are supported"}
