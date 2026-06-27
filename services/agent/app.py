import base64
import io
import json
import logging
import os
import time
from contextvars import ContextVar
from typing import Optional
from langchain_core.rate_limiters import InMemoryRateLimiter
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("langchain").setLevel(logging.DEBUG)
logging.getLogger("langchain_core").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel


YOLO_SERVICE_URL = os.environ.get("YOLO_SERVICE_URL", "http://localhost:8080")
MODEL = os.environ.get("MODEL")

# Text-only models
ALLOWED_MODELS = {
    "openai:gpt-5.4-mini",
    "anthropic:claude-haiku-4-5",
    "google_genai:gemini-1.5-flash",
}

llm_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # 30 requests per minute
    check_every_n_seconds=0.1,
    max_bucket_size=5,
)

if MODEL not in ALLOWED_MODELS:
    allowed_list = "\n  ".join(sorted(ALLOWED_MODELS))
    raise SystemExit(
        f"\n[ERROR] MODEL='{MODEL}' is not allowed.\n"
        f"Set MODEL in your .env to one of the supported text-only models:\n  {allowed_list}\n"
    )

_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

SYSTEM_PROMPT = (
    "You are an AI vision assistant. You help users understand and analyze images. "
    "Use the available tools to extract information from images. "
)

_current_image_b64: ContextVar[Optional[str]] = ContextVar("current_image_b64", default=None)
# Holds a mutable dict so the tool can write back through LangChain's context copy boundary.
# LangChain's invoke() uses copy_context().run(), which isolates ContextVar *assignments*
# but not mutations to objects already referenced by the var.
_result_store: ContextVar[Optional[dict]] = ContextVar("result_store", default=None)

@tool
def detect_objects() -> str:
    """Detect and identify objects in the image provided by the user using YOLO object detection."""
    image_b64 = _current_image_b64.get()
    if not image_b64:
        return json.dumps({"error": "No image was provided by the user."})

    image_bytes = base64.b64decode(image_b64)
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{YOLO_SERVICE_URL}/predict",
            files={"file": ("image.jpg", io.BytesIO(image_bytes), "image/jpeg")},
        )
        response.raise_for_status()
        result = response.json()

    store = _result_store.get()
    if store is not None:
        store["prediction_uid"] = result.get("uid")

    return json.dumps(result)


@tool
def get_annotated_image() -> str:
    """Return the annotated image with bounding boxes from the most recent detection.
    Only call this when the user explicitly asks to see the annotated image or bounding boxes."""
    store = _result_store.get()
    uid = store.get("prediction_uid") if store else None
    if not uid:
        return json.dumps({"error": "No detection has been run yet. Use detect_objects first."})

    with httpx.Client(timeout=30.0) as client:
        img_response = client.get(f"{YOLO_SERVICE_URL}/prediction/{uid}/image")
        if img_response.status_code == 200:
            if store is not None:
                store["annotated_image_b64"] = base64.b64encode(img_response.content).decode()
            return json.dumps({"success": True})
        return json.dumps({"error": f"Could not retrieve image (status {img_response.status_code})"})


# Registry: map tool name -> tool function
TOOLS = {
    detect_objects.name: detect_objects,
    get_annotated_image.name: get_annotated_image,
}

llm = init_chat_model(MODEL, temperature=0, rate_limiter=llm_rate_limiter)
llm_with_tools = llm.bind_tools(list(TOOLS.values()))

_profile = llm.profile
if not _profile.get("tool_calling"):
    raise SystemExit(
        f"\n[ERROR] Model '{MODEL}' does not support tool calling according to its profile.\n"
        "This agent requires tool calling. Choose a model with tool_calling=True.\n"
    )
_max_input_tokens: int = _profile.get("max_input_tokens", 0)
logger.info("Model profile loaded: model=%s, max_input_tokens=%s", MODEL, _max_input_tokens)


def run_agent(history: list, max_iterations: int = 10) -> dict:
    """
    Simple ReAct loop:
      1. Send messages to the LLM.
      2. If the LLM requests tool calls, execute them and append results.
      3. Repeat until the LLM returns a plain text response.
    Returns a dict with response text and loop metrics.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history
    tools_called: list[str] = []
    input_tokens_total: int = 0
    output_tokens_total: int = 0

    for iteration in range(1, max_iterations + 1):
        response: AIMessage = llm_with_tools.invoke(messages)

        usage = response.usage_metadata or {}
        input_tokens_total += usage.get("input_tokens", 0)
        output_tokens_total += usage.get("output_tokens", 0)

        if _max_input_tokens > 0 and input_tokens_total >= 0.9 * _max_input_tokens:
            logger.warning(
                "Approaching context limit: cumulative input_tokens=%d >= 90%% of max_input_tokens=%d",
                input_tokens_total,
                _max_input_tokens,
            )

        messages.append(response)

        if not response.tool_calls:
            return {
                "response": response.content,
                "iterations": iteration,
                "tools_called": tools_called,
                "context_limit_exceeded": False,
                "tokens_used": {
                    "input": input_tokens_total,
                    "output": output_tokens_total,
                    "total": input_tokens_total + output_tokens_total,
                },
            }

        for tool_call in response.tool_calls:
            tools_called.append(tool_call["name"])
            tool_fn = TOOLS[tool_call["name"]]
            messages.append(tool_fn.invoke(tool_call))

    return {
        "response": "",
        "iterations": max_iterations,
        "tools_called": tools_called,
        "context_limit_exceeded": True,
        "tokens_used": {
            "input": input_tokens_total,
            "output": output_tokens_total,
            "total": input_tokens_total + output_tokens_total,
        },
    }


app = FastAPI(title="Vision Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


class ChatMessage(BaseModel):
    role: str                           # "user" or "assistant"
    content: str
    image_base64: Optional[str] = None  # only on user messages that carry an image


class ChatRequest(BaseModel):
    messages: list[ChatMessage]         # full conversation thread, oldest first


class ChatResponse(BaseModel):
    response: str
    prediction_id: Optional[str] = None
    annotated_image: Optional[str] = None
    agent_loop_time_s: float
    iterations: int
    tools_called: list[str]
    context_limit_exceeded: bool
    tokens_used: dict  # {"input": N, "output": N, "total": N}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    lc_messages = []
    latest_image = None

    for msg in request.messages:
        if msg.role == "user":
            if msg.image_base64:
                latest_image = msg.image_base64
                content = msg.content + "\n[An image was uploaded. Use existing tools to analyze it according to user instructions.]"
            else:
                content = msg.content
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=msg.content))

    result_store: dict = {}
    image_token = _current_image_b64.set(latest_image)
    store_token = _result_store.set(result_store)
    try:
        start = time.time()
        agent_result = run_agent(lc_messages)
        elapsed = round(time.time() - start, 2)
        return ChatResponse(
            response=agent_result["response"],
            prediction_id=result_store.get("prediction_uid"),
            annotated_image=result_store.get("annotated_image_b64"),
            agent_loop_time_s=elapsed,
            iterations=agent_result["iterations"],
            tools_called=agent_result["tools_called"],
            context_limit_exceeded=agent_result["context_limit_exceeded"],
            tokens_used=agent_result["tokens_used"],
        )
    finally:
        _current_image_b64.reset(image_token)
        _result_store.reset(store_token)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
