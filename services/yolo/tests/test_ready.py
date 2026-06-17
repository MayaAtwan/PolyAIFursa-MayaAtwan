import app as app_module


def test_ready_returns_200_when_not_shutting_down(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_returns_503_when_shutting_down(client, monkeypatch):
    monkeypatch.setattr(app_module, "is_shutting_down", True)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == "Service is shutting down"
