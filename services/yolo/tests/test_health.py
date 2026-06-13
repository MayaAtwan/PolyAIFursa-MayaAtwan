# Tests for GET /health
#
# What we check: the endpoint is reachable and returns {"status": "ok"}.
# How we test it: plain GET request via TestClient — no DB or model needed.


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
