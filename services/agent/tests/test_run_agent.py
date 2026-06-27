"""Unit tests for the run_agent() ReAct loop.

Following the LangChain unit-testing approach, the LLM is replaced with
FakeMessagesListChatModel, which returns predefined AIMessages (preserving
tool_calls and usage_metadata). No real LLM or YOLO service is contacted.
"""
import base64

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage


def _tool_call(name="detect_objects", call_id="call_1"):
    return {"name": name, "args": {}, "id": call_id, "type": "tool_call"}


def _set_llm(agent_app, monkeypatch, responses):
    """Swap the module-level llm_with_tools for a fake that replays `responses`."""
    monkeypatch.setattr(agent_app, "llm_with_tools", FakeMessagesListChatModel(responses=responses))


# ── Fake YOLO HTTP client (for the detect_objects tool) ────────────────────────

class _FakeHTTPResponse:
    def __init__(self, json_body):
        self._json = json_body
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class _FakeHTTPClient:
    def __init__(self, json_body):
        self._json = json_body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, files=None):
        return _FakeHTTPResponse(self._json)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_no_tool_calls(agent_app, monkeypatch):
    _set_llm(
        agent_app,
        monkeypatch,
        [AIMessage(content="Hello", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})],
    )

    result = agent_app.run_agent([HumanMessage(content="hi")])

    assert result["response"] == "Hello"
    assert result["iterations"] == 1
    assert result["tools_called"] == []
    assert result["context_limit_exceeded"] is False
    assert result["tokens_used"] == {"input": 10, "output": 5, "total": 15}


def test_executes_tool_then_responds(agent_app, monkeypatch):
    # First the LLM asks for a tool, then it produces a final answer.
    _set_llm(
        agent_app,
        monkeypatch,
        [
            AIMessage(content="", tool_calls=[_tool_call()]),
            AIMessage(content="I see a person."),
        ],
    )
    # Provide an image so detect_objects proceeds, and stub the YOLO HTTP call.
    agent_app._current_image_b64.set(base64.b64encode(b"img").decode())
    agent_app._result_store.set({})
    monkeypatch.setattr(
        agent_app.httpx, "Client", lambda *a, **k: _FakeHTTPClient({"uid": "abc-123", "detection_objects": []})
    )

    result = agent_app.run_agent([HumanMessage(content="what is in this image")])

    assert result["response"] == "I see a person."
    assert result["iterations"] == 2
    assert result["tools_called"] == ["detect_objects"]
    assert result["context_limit_exceeded"] is False
    # The tool wrote the prediction uid back into the result store.
    assert agent_app._result_store.get()["prediction_uid"] == "abc-123"


def test_tool_with_no_image(agent_app, monkeypatch):
    """When no image is set, detect_objects returns an error without any HTTP call."""
    _set_llm(
        agent_app,
        monkeypatch,
        [
            AIMessage(content="", tool_calls=[_tool_call()]),
            AIMessage(content="No image was available."),
        ],
    )
    agent_app._current_image_b64.set(None)
    agent_app._result_store.set({})

    # If an HTTP call were made it would raise, proving the no-image short-circuit.
    def _boom(*a, **k):
        raise AssertionError("detect_objects must not make an HTTP call without an image")

    monkeypatch.setattr(agent_app.httpx, "Client", _boom)

    result = agent_app.run_agent([HumanMessage(content="describe it")])

    assert result["tools_called"] == ["detect_objects"]
    assert result["response"] == "No image was available."
    assert result["iterations"] == 2


def test_max_iterations_exceeded(agent_app, monkeypatch):
    # Every response asks for a tool, so the loop never terminates on its own.
    _set_llm(agent_app, monkeypatch, [AIMessage(content="", tool_calls=[_tool_call()])])
    agent_app._current_image_b64.set(None)  # no-image path -> no HTTP
    agent_app._result_store.set({})

    result = agent_app.run_agent([HumanMessage(content="loop")], max_iterations=3)

    assert result["response"] == ""
    assert result["iterations"] == 3
    assert result["context_limit_exceeded"] is True
    assert result["tools_called"] == ["detect_objects", "detect_objects", "detect_objects"]


def test_two_tool_calls_in_one_turn(agent_app, monkeypatch):
    """A single AIMessage may request multiple tools; all must be executed and fed back."""
    _set_llm(
        agent_app,
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(name="detect_objects", call_id="c1"),
                    _tool_call(name="get_annotated_image", call_id="c2"),
                ],
            ),
            AIMessage(content="done"),
        ],
    )
    agent_app._current_image_b64.set(None)  # both tools short-circuit without HTTP
    agent_app._result_store.set({})

    result = agent_app.run_agent([HumanMessage(content="hi")])

    assert result["tools_called"] == ["detect_objects", "get_annotated_image"]
    assert result["iterations"] == 2
    assert result["response"] == "done"


def test_token_accounting(agent_app, monkeypatch):
    """Token usage is summed across every LLM call in the loop."""
    _set_llm(
        agent_app,
        monkeypatch,
        [
            AIMessage(content="", tool_calls=[_tool_call()], usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}),
            AIMessage(content="done", usage_metadata={"input_tokens": 20, "output_tokens": 6, "total_tokens": 26}),
        ],
    )
    agent_app._current_image_b64.set(None)  # no-image path -> no HTTP
    agent_app._result_store.set({})

    result = agent_app.run_agent([HumanMessage(content="hi")])

    assert result["tokens_used"] == {"input": 30, "output": 10, "total": 40}
