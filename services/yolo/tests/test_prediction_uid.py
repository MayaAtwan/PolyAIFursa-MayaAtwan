# Tests for GET /prediction/{uid} and GET /prediction/{uid}/image
#
# What we check:
#   - Fetching a session by uid returns the full record with nested detection objects.
#   - Fetching a non-existent uid returns 404.
#   - Fetching the annotated image returns the correct file bytes.
#   - Fetching an image when the uid is missing or the file was deleted returns 404.
#
# How we test it:
#   - Rows are inserted directly via save_prediction_session / save_detection_object
#     (no HTTP round-trip needed for setup).
#   - The image-download test writes a real file to tmp_path so FileResponse has
#     something to serve.

from tests.conftest import save_detection_object, save_prediction_session


def test_get_prediction_by_uid_returns_prediction(client):
    save_prediction_session(
        "abc-123",
        "uploads/original/abc-123.jpg",
        "uploads/predicted/abc-123.jpg",
    )
    save_detection_object("abc-123", "person", 0.91, [10, 20, 100, 200])
    save_detection_object("abc-123", "car", 0.50, [1, 2, 3, 4])

    response = client.get("/prediction/abc-123")

    assert response.status_code == 200

    data = response.json()
    assert data["uid"] == "abc-123"
    assert "timestamp" in data
    assert data["original_image"] == "uploads/original/abc-123.jpg"
    assert data["predicted_image"] == "uploads/predicted/abc-123.jpg"
    assert len(data["detection_objects"]) == 2

    labels = [obj["label"] for obj in data["detection_objects"]]
    assert "person" in labels
    assert "car" in labels


def test_get_prediction_by_uid_returns_404_when_not_found(client):
    response = client.get("/prediction/not-found")

    assert response.status_code == 404
    assert response.json() == {"detail": "Prediction not found"}


def test_get_prediction_image_returns_file(client, tmp_path):
    predicted_image = tmp_path / "predicted.jpg"
    predicted_image.write_bytes(b"fake image content")

    save_prediction_session(
        "img-123",
        "uploads/original/img-123.jpg",
        str(predicted_image),
    )

    response = client.get("/prediction/img-123/image")

    assert response.status_code == 200
    assert response.content == b"fake image content"


def test_get_prediction_image_returns_404_when_uid_not_found(client):
    response = client.get("/prediction/missing/image")

    assert response.status_code == 404
    assert response.json() == {"detail": "Image not found"}


def test_get_prediction_image_returns_404_when_file_missing(client):
    save_prediction_session(
        "missing-file",
        "uploads/original/missing-file.jpg",
        "uploads/predicted/missing-file.jpg",
    )

    response = client.get("/prediction/missing-file/image")

    assert response.status_code == 404
    assert response.json() == {"detail": "Image not found"}
