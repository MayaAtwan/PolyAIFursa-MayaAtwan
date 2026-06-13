# Tests for GET /predictions/label/{label} and GET /predictions/label/
#
# What we check:
#   - A matching label returns only the sessions that contain that label,
#     with only the matching detection objects nested inside.
#   - A label with no matches returns an empty list (not 404).
#   - An empty path segment (/predictions/label/) returns 400.
#   - A whitespace-only label (%20) is treated as empty and returns 400.
#
# How we test it:
#   - The seeded_db fixture (from conftest) pre-loads two sessions:
#       abc-123: person (0.91) + car (0.50)
#       def-456: dog (0.49)
#   - Tests query by label and assert on the returned structure.


def test_get_predictions_by_label_returns_matching_sessions(client, seeded_db):
    response = client.get("/predictions/label/person")

    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1

    session = data[0]
    assert session["uid"] == "abc-123"
    assert "timestamp" in session
    assert len(session["detection_objects"]) == 1
    assert session["detection_objects"][0]["label"] == "person"
    assert session["detection_objects"][0]["score"] == 0.91
    assert session["detection_objects"][0]["box"] == "[10, 20, 100, 200]"


def test_get_predictions_by_label_returns_empty_list_when_no_match(client, seeded_db):
    response = client.get("/predictions/label/cat")

    assert response.status_code == 200
    assert response.json() == []


def test_get_predictions_by_empty_label_returns_400(client):
    response = client.get("/predictions/label/")

    assert response.status_code == 400
    assert response.json() == {"detail": "Label cannot be empty"}


def test_get_predictions_by_whitespace_label_returns_400(client):
    response = client.get("/predictions/label/%20")

    assert response.status_code == 400
    assert response.json() == {"detail": "Label cannot be empty"}
