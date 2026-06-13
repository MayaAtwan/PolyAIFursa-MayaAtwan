# Tests for GET /predictions/score/{min_score}
#
# What we check:
#   - Returns only detection objects with score >= min_score.
#   - Returns an empty list when nothing meets the threshold.
#   - min_score=0.0 returns every object in the DB.
#   - Scores outside [0.0, 1.0] return 400.
#
# How we test it:
#   - The seeded_db fixture (from conftest) inserts three objects:
#       person: 0.91, car: 0.50, dog: 0.49
#   - Boundary conditions are tested with 0.5, 0.0, 1.0, -0.1, 1.1.


def test_get_predictions_by_score_returns_objects_above_or_equal_score(client, seeded_db):
    response = client.get("/predictions/score/0.5")

    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2

    labels = [obj["label"] for obj in data]
    assert "person" in labels
    assert "car" in labels
    assert "dog" not in labels

    for obj in data:
        assert obj["score"] >= 0.5
        assert "id" in obj
        assert "prediction_uid" in obj
        assert "label" in obj
        assert "score" in obj
        assert "box" in obj


def test_get_predictions_by_score_returns_empty_list_when_no_match(client, seeded_db):
    response = client.get("/predictions/score/1.0")

    assert response.status_code == 200
    assert response.json() == []


def test_get_predictions_by_score_accepts_zero(client, seeded_db):
    response = client.get("/predictions/score/0.0")

    assert response.status_code == 200
    assert len(response.json()) == 3


def test_get_predictions_by_score_rejects_score_below_zero(client):
    response = client.get("/predictions/score/-0.1")

    assert response.status_code == 400
    assert response.json() == {"detail": "min_score must be between 0.0 and 1.0"}


def test_get_predictions_by_score_rejects_score_above_one(client):
    response = client.get("/predictions/score/1.1")

    assert response.status_code == 400
    assert response.json() == {"detail": "min_score must be between 0.0 and 1.0"}
