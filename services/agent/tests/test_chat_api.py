"""API-layer unit tests for the agent endpoints.

These mock the entire agentic loop (run_agent) so the API layer is exercised
in isolation — no LLM and no YOLO service are involved.
"""


def _fixed_agent_result(**overrides):
    result = {
        "response": "I found a person in the image.",
        "iterations": 2,
        "tools_called": ["detect_objects"],
        "context_limit_exceeded": False,
    }
    result.update(overrides)
    return result


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_agent_result(client, agent_app, monkeypatch):
    monkeypatch.setattr(agent_app, "run_agent", lambda history: _fixed_agent_result())

    response = client.post("/chat", json={"messages": [{"role": "user", "content": "What is in this image?"}]})

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "I found a person in the image."
    assert body["iterations"] == 2
    assert body["tools_called"] == ["detect_objects"]
    assert body["context_limit_exceeded"] is False
    assert isinstance(body["agent_loop_time_s"], (int, float))


def test_chat_populates_prediction_and_image(client, agent_app, monkeypatch):
    """The endpoint should surface values the agentic loop writes into the result store."""

    def fake_run_agent(history):
        store = agent_app._result_store.get()
        store["prediction_uid"] = "abc-123"
        store["annotated_image_b64"] = "ZmFrZS1pbWFnZQ=="
        return _fixed_agent_result()

    monkeypatch.setattr(agent_app, "run_agent", fake_run_agent)

    response = client.post("/chat", json={"messages": [{"role": "user", "content": "Annotate it"}]})

    assert response.status_code == 200
    body = response.json()
    assert body["prediction_id"] == "abc-123"
    assert body["annotated_image"] == "ZmFrZS1pbWFnZQ=="


def test_chat_image_message(client, agent_app, monkeypatch):
    """A user message carrying an image is accepted and the image is threaded to the loop."""
    captured = {}

    def fake_run_agent(history):
        captured["image"] = agent_app._current_image_b64.get()
        return _fixed_agent_result()

    monkeypatch.setattr(agent_app, "run_agent", fake_run_agent)

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Look", "image_base64": "aW1hZ2U="}]},
    )

    assert response.status_code == 200
    assert captured["image"] == "aW1hZ2U="


def test_chat_validation_error(client):
    response = client.post("/chat", json={"not_messages": []})
    assert response.status_code == 422
