# Tests for POST /predict
#
# What we check:
#   - Happy path: a valid S3 key triggers download + detection and returns structured data.
#   - Validation: a key with a non-image extension is rejected with 400 before any
#     S3 download or model inference happens.
#
# How we test it:
#   - The real YOLO model is replaced with FakeModel (from conftest) via monkeypatch.
#   - The S3 helpers are faked in conftest: an "/original/" key downloads a default
#     in-memory JPEG, and uploads are captured in s3_store.

import app as app_module
from tests.conftest import FakeModel


def test_predict_returns_prediction_data(client, monkeypatch, s3_store):
    monkeypatch.setattr(app_module, "model", FakeModel())

    response = client.post(
        "/predict",
        json={"image_s3_key": "chat-1/pred-1/original/test.jpg"},
    )

    assert response.status_code == 200

    data = response.json()
    assert "uid" in data
    assert data["original_image"] == "chat-1/pred-1/original/test.jpg"
    assert data["predicted_image"] == "chat-1/pred-1/predicted/test.jpg"
    assert len(data["detection_objects"]) == 1
    assert data["detection_objects"][0]["label"] == "person"
    # The annotated image was uploaded back to S3 under the predicted key.
    assert "chat-1/pred-1/predicted/test.jpg" in s3_store


def test_predict_rejects_non_image_file(client):
    response = client.post(
        "/predict",
        json={"image_s3_key": "chat-1/pred-1/original/document.pdf"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Only image files are supported"}
