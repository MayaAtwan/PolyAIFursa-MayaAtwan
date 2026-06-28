# Tests for GET /prediction/{uid} and GET /prediction/{uid}/image
#
# What we check:
#   - Fetching a session by uid returns the full record with nested detection objects.
#   - Fetching a non-existent uid returns 404.
#   - Fetching the annotated image returns the correct bytes (from the fake S3 store).
#   - Fetching an image when the uid is missing, or the object is absent from S3,
#     returns 404.
#
# How we test it:
#   - Rows are inserted directly via the ORM seed helpers (no HTTP round-trip needed).
#   - The image-download test seeds the fake S3 store (s3_store) so the endpoint has
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


def test_get_prediction_image_returns_file(client, s3_store):
    predicted_key = "chat-x/pred-x/predicted/img-123.jpg"
    s3_store[predicted_key] = b"fake image content"

    save_prediction_session(
        "img-123",
        "chat-x/pred-x/original/img-123.jpg",
        predicted_key,
    )

    response = client.get("/prediction/img-123/image")

    assert response.status_code == 200
    assert response.content == b"fake image content"


def test_get_prediction_image_returns_404_when_uid_not_found(client):
    response = client.get("/prediction/missing/image")

    assert response.status_code == 404
    assert response.json() == {"detail": "Image not found"}


def test_get_prediction_image_returns_404_when_object_missing(client):
    # Session exists but its predicted object was never uploaded to S3.
    save_prediction_session(
        "missing-file",
        "chat-y/pred-y/original/missing-file.jpg",
        "chat-y/pred-y/predicted/missing-file.jpg",
    )

    response = client.get("/prediction/missing-file/image")

    assert response.status_code == 404
    assert response.json() == {"detail": "Image not found"}
