import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Make the service's app.py importable (parent of this tests/ directory),
# mirroring how the YOLO service test suite imports `app`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# app.py does real work at import time:
#   - exits unless MODEL is an allowed value
#   - builds a real LLM client via init_chat_model() and reads llm.profile
# Set a valid MODEL and fake init_chat_model BEFORE importing app so no real
# LLM client (and no API key / network) is ever required.
os.environ["MODEL"] = "anthropic:claude-haiku-4-5"

_fake_llm = MagicMock()
_fake_llm.profile = {"tool_calling": True, "max_input_tokens": 200000}
_fake_llm.bind_tools.return_value = MagicMock()

with patch("langchain.chat_models.init_chat_model", return_value=_fake_llm):
    import app as agent_app_module  # noqa: E402


@pytest.fixture
def agent_app():
    """The imported agent app module (for monkeypatching run_agent / llm_with_tools)."""
    return agent_app_module


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    return TestClient(agent_app_module.app)


@pytest.fixture(autouse=True)
def reset_context_vars():
    """Ensure tool ContextVars don't leak between tests."""
    yield
    agent_app_module._current_image_b64.set(None)
    agent_app_module._result_store.set(None)
