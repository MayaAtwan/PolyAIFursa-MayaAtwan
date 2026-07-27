import base64
import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
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
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from PIL import Image as PILImage
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from s3_utils import AWS_S3_BUCKET, upload_bytes, generate_presigned_url, save_session_image, load_session_image

YOLO_SERVICE_URL = os.environ.get("YOLO_SERVICE_URL", "http://localhost:8080")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:9000/mcp")
MODEL = os.environ.get("MODEL")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

if not AWS_S3_BUCKET:
    logger.warning(
        "AWS_S3_BUCKET is not set; image uploads to S3 will fail. "
        "Set AWS_S3_BUCKET (and AWS_REGION) in the environment."
    )

# Text-only models
ALLOWED_MODELS = {
    "bedrock_converse:anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock_converse:amazon.nova-micro-v1:0",
    "bedrock_converse:amazon.nova-lite-v1:0",
    "bedrock_converse:openai.gpt-oss-20b-1:0",
    "bedrock_converse:meta.llama3-1-8b-instruct-v1:0",
    "bedrock_converse:mistral.mistral-7b-instruct-v0:2",
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
    "When applying any image transformation (rotate, flip, blur, resize, crop, add_noise): "
    "- For the full uploaded image: call the transformation tool directly, do NOT provide image_b64. "
    "- For a specific detected object: call detect_objects, then get_object_region_b64 to select the region, "
    "then call the transformation tool directly without providing image_b64. "
    "The image or region is always supplied to the transformation tool automatically. "
    "IMPORTANT: Every transformation always operates on the latest version of the image — "
    "whether the request comes in the same message or a later message in the conversation. "
    "Never revert to the original unless the user explicitly asks to. "
    "When the user requests multiple transformations, call them sequentially — "
    "each result automatically becomes the input to the next. "
    "After applying any transformation (flip, rotate, resize, or crop), call detect_objects again "
    "before selecting a region — the previous detection is stale. "
    "If and ONLY if the user explicitly asks to go back to the original image or undo all transformations, "
    "call reset_to_original. Do not call reset_to_original for any other reason. "
    "For resize, prefer scale_factor (e.g. 0.5 to halve, 2.0 to double) over absolute pixel dimensions, "
    "especially when resizing a detected object region."
)

_current_image_b64: ContextVar[Optional[str]] = ContextVar("current_image_b64", default=None)
_current_chat_id: ContextVar[Optional[str]] = ContextVar("current_chat_id", default=None)
# Holds a mutable dict so the tool can write back through LangChain's context copy boundary.
# LangChain's invoke() uses copy_context().run(), which isolates ContextVar *assignments*
# but not mutations to objects already referenced by the var.
_result_store: ContextVar[Optional[dict]] = ContextVar("result_store", default=None)


@tool
def detect_objects() -> str:
    """Detect and identify objects in the image provided by the user using YOLO object detection."""
    store = _result_store.get()
    image_b64 = (store.get("last_transformed_b64") if store else None) or _current_image_b64.get()
    if not image_b64:
        return json.dumps({"error": "No image was provided by the user."})

    image_bytes = base64.b64decode(image_b64)

    # Upload the original image to S3, organised per chat:
    #   <chat_id>/<prediction_id>/original/<image_name>
    # Then hand YOLO only the object key — not the image bytes.
    chat_id = _current_chat_id.get() or str(uuid.uuid4())
    prediction_id = str(uuid.uuid4())
    image_s3_key = f"{chat_id}/{prediction_id}/original/image.jpg"
    upload_bytes(image_bytes, image_s3_key)

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{YOLO_SERVICE_URL}/predict",
            json={"image_s3_key": image_s3_key},
        )
        response.raise_for_status()
        result = response.json()

    store = _result_store.get()
    if store is not None:
        store["prediction_uid"] = result.get("uid")
        store["detection_objects"] = result.get("detection_objects", [])

    return json.dumps(result)


@tool
def get_annotated_image() -> str:
    """Return the annotated image with bounding boxes from the most recent detection.
    Only call this when the user explicitly asks to see the annotated image or bounding boxes."""
    store = _result_store.get()
    uid = store.get("prediction_uid") if store else None
    if not uid:
        return json.dumps({"error": "No detection has been run yet. Use detect_objects first."})

    chat_id = _current_chat_id.get() or str(uuid.uuid4())
    with httpx.Client(timeout=30.0) as client:
        img_response = client.get(f"{YOLO_SERVICE_URL}/prediction/{uid}/image")
        if img_response.status_code == 200:
            annotated_s3_key = f"{chat_id}/{uid}/annotated/image.png"
            upload_bytes(img_response.content, annotated_s3_key, content_type="image/png")
            if store is not None:
                store["annotated_image_key"] = annotated_s3_key
            return json.dumps({"success": True})
        return json.dumps({"error": f"Could not retrieve image (status {img_response.status_code})"})


@tool
def get_object_region_b64(label: str, rank: int) -> str:
    """Select a detected object's region as the source for the next image processing operation.
    label: object class (e.g. 'dog'). rank: 1=rightmost, 2=second-from-right, etc.
    Call detect_objects first. After calling this, call the image processing tool directly — the region is passed automatically."""
    store = _result_store.get()
    image_b64 = (store.get("last_transformed_b64") if store else None) or _current_image_b64.get()
    if not store or not image_b64:
        return json.dumps({"error": "No image or detection results available. Call detect_objects first."})

    detections = store.get("detection_objects", [])
    matches = sorted(
        [d for d in detections if d["label"] == label],
        key=lambda d: d["box"][0],
        reverse=True,
    )
    if rank < 1 or rank > len(matches):
        return json.dumps({"error": f"rank {rank} out of range; found {len(matches)} '{label}' object(s)"})

    x1, y1, x2, y2 = [int(v) for v in matches[rank - 1]["box"]]
    img = PILImage.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    buf = io.BytesIO() #create an in-memory bytes buffer to hold the cropped image
    img.crop((x1, y1, x2, y2)).save(buf, format="JPEG")
    region_b64 = base64.b64encode(buf.getvalue()).decode()
    if store is not None:
        store["last_region_box"] = [x1, y1, x2, y2]
        store["pending_image_b64"] = region_b64
    return json.dumps({"success": True, "label": label, "rank": rank})


@tool
def upload_result_image() -> str:
    """Upload the current image from the chat request. Do not provide image data."""
    store = _result_store.get()
    image_b64 = (store.get("last_transformed_b64") if store else None) or _current_image_b64.get()

    if not image_b64:
        return "No image was uploaded in this chat request."

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        return "Could not upload image: invalid image data."

    chat_id = _current_chat_id.get() or str(uuid.uuid4())
    key = f"{chat_id}/{uuid.uuid4()}/processed/image.jpg"
    upload_bytes(image_bytes, key)
    store = _result_store.get()
    if store is not None:
        store["processed_image_key"] = key
    return json.dumps({"success": True})


@tool
def reset_to_original() -> str:
    """Revert the working image to the original uploaded image, discarding all transforms."""
    chat_id = _current_chat_id.get()
    store = _result_store.get() or {}
    original = load_session_image(chat_id, "original") if chat_id else None
    if original is None:
        original = _current_image_b64.get()
    if original:
        store["last_transformed_b64"] = original
        if chat_id:
            save_session_image(chat_id, "latest", original)
        return json.dumps({"success": True, "message": "Reverted to original image."})
    return json.dumps({"success": False, "message": "No original image found."})


# Registry: map tool name -> tool function (MCP tools added at startup via lifespan)
TOOLS = {
    detect_objects.name: detect_objects,
    get_annotated_image.name: get_annotated_image,
    get_object_region_b64.name: get_object_region_b64,
    reset_to_original.name: reset_to_original,
}

# region_name is a Bedrock-only kwarg; other providers reject it, so add it conditionally.
init_kwargs = {"temperature": 0, "rate_limiter": llm_rate_limiter}
if MODEL.startswith("bedrock"):
    init_kwargs["region_name"] = AWS_REGION
llm = init_chat_model(MODEL, **init_kwargs)
llm_with_tools = llm.bind_tools(list(TOOLS.values()))

_profile = llm.profile
# Some providers (e.g. langchain-aws / Bedrock) don't publish a capability profile,
# so it comes back empty. Only enforce the tool-calling guard when the profile actually
# reports capabilities; otherwise we can't determine support and just warn.
if not _profile:
    logger.warning(
        "Model '%s' has no capability profile; skipping tool-calling check. "
        "Ensure the selected model supports tool calling.",
        MODEL,
    )
elif not _profile.get("tool_calling"):
    raise SystemExit(
        f"\n[ERROR] Model '{MODEL}' does not support tool calling according to its profile.\n"
        "This agent requires tool calling. Choose a model with tool_calling=True.\n"
    )
_max_input_tokens: int = _profile.get("max_input_tokens", 0)
logger.info("Model profile loaded: model=%s, max_input_tokens=%s", MODEL, _max_input_tokens)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_with_tools
    try:
        mcp_client = MultiServerMCPClient(
            {"img-proc": {"url": MCP_SERVER_URL, "transport": "sse"}}
        )
        mcp_tools = await mcp_client.get_tools()
        for t in mcp_tools:
            TOOLS[t.name] = t
        llm_with_tools = llm.bind_tools(list(TOOLS.values()))
        logger.info("MCP tools loaded: %s", [t.name for t in mcp_tools])
    except Exception as e:
        logger.warning("MCP server unavailable, starting without image-processing tools: %s", e)
    yield


def message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(item["text"])
                elif item.get("text"):
                    parts.append(item["text"])
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _composite_region(original_b64: str, region_b64: str, box: list) -> str:
    """Paste a processed region back into the original image at the given bounding box."""
    x1, y1, x2, y2 = box
    original = PILImage.open(io.BytesIO(base64.b64decode(original_b64))).convert("RGB")
    region = PILImage.open(io.BytesIO(base64.b64decode(region_b64))).convert("RGB")
    region = region.resize((x2 - x1, y2 - y1), PILImage.LANCZOS)
    original.paste(region, (x1, y1))
    buf = io.BytesIO()
    original.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _composite_region_centered(original_b64: str, region_b64: str, box: list) -> str:
    """Paste a resized region centered on the original bounding box, preserving surrounding background."""
    x1, y1, x2, y2 = box
    original = PILImage.open(io.BytesIO(base64.b64decode(original_b64))).convert("RGB")
    region = PILImage.open(io.BytesIO(base64.b64decode(region_b64))).convert("RGB")
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rw, rh = region.size
    # Clamp so the pasted region never goes outside the canvas boundaries.
    paste_x = max(0, min(cx - rw // 2, original.width - rw))
    paste_y = max(0, min(cy - rh // 2, original.height - rh))
    original.paste(region, (paste_x, paste_y))
    buf = io.BytesIO()
    original.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


_IMG_PROC_MCP_TOOLS = {"rotate", "flip", "blur", "resize", "crop", "add_noise"}


async def run_agent(history: list, max_iterations: int = 10) -> dict:
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
        response: AIMessage = await llm_with_tools.ainvoke(messages)

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
                "response": message_content_to_text(response.content),
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

            if tool_call["name"] in _IMG_PROC_MCP_TOOLS:
                store = _result_store.get() or {}
                source_b64 = store.pop("pending_image_b64", None) or store.get("last_transformed_b64") or _current_image_b64.get() or ""
                tool_call = {**tool_call, "args": {**tool_call["args"], "image_b64": source_b64}}

            tool_msg = await tool_fn.ainvoke(tool_call)

            if tool_call["name"] in _IMG_PROC_MCP_TOOLS:
                content = tool_msg.content
                if isinstance(content, list) and content:
                    first = content[0]
                    result_b64 = first.get("text", "") if isinstance(first, dict) else str(first)
                elif isinstance(content, str):
                    result_b64 = content
                else:
                    result_b64 = None
                if result_b64:
                    try:
                        base64.b64decode(result_b64, validate=True)
                    except Exception:
                        logger.warning("MCP tool '%s' returned non-image content, skipping upload", tool_call["name"])
                        result_b64 = None
                if result_b64:
                    store = _result_store.get()
                    if store is None:
                        store = {}
                    region_box = store.pop("last_region_box", None)
                    if region_box:
                        original_b64 = store.get("last_transformed_b64") or _current_image_b64.get()
                        if original_b64:
                            if tool_call["name"] == "resize":
                                result_b64 = _composite_region_centered(original_b64, result_b64, region_box)
                            elif tool_call["name"] != "crop":
                                result_b64 = _composite_region(original_b64, result_b64, region_box)
                    store["last_transformed_b64"] = result_b64
                    _cid = _current_chat_id.get()
                    if _cid:
                        try:
                            save_session_image(_cid, "latest", result_b64)
                        except Exception:
                            logger.warning("Could not save session latest to S3 for chat_id=%s", _cid)
                    await upload_result_image.ainvoke({})
                tool_msg.content = json.dumps({"success": True})

            messages.append(tool_msg)

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


app = FastAPI(title="Vision Agent", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "You're sending requests too quickly. Please wait a moment and try again."},
    )


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
    chat_id: Optional[str] = None       # groups S3 objects per chat; generated if absent


class ChatResponse(BaseModel):
    response: str
    prediction_id: Optional[str] = None
    annotated_image_url: Optional[str] = None
    agent_loop_time_s: float
    iterations: int
    tools_called: list[str]
    context_limit_exceeded: bool
    tokens_used: dict  # {"input": N, "output": N, "total": N}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, request_body: ChatRequest):
    lc_messages = []
    latest_image = None

    for msg in request_body.messages:
        if msg.role == "user":
            if msg.image_base64:
                latest_image = msg.image_base64
                content = msg.content + "\n[An image was uploaded. Use existing tools to analyze it according to user instructions.]"
            else:
                content = msg.content
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=msg.content))

    chat_id = request_body.chat_id or str(uuid.uuid4())

    # Detect whether an image was uploaded in this specific turn (last user message),
    # as opposed to images that appear only in earlier turns of the conversation history.
    last_msg = request_body.messages[-1] if request_body.messages else None
    new_image_this_turn = last_msg.image_base64 if (last_msg and last_msg.role == "user") else None

    result_store: dict = {}

    if new_image_this_turn:
        # New upload: persist as the session original so it can be restored later.
        try:
            save_session_image(chat_id, "original", new_image_this_turn)
        except Exception:
            logger.warning("Could not save session original to S3 for chat_id=%s", chat_id)
    else:
        # No new image this turn: restore the latest transformed image from the previous turn.
        try:
            session_latest = load_session_image(chat_id, "latest")
            if session_latest:
                result_store["last_transformed_b64"] = session_latest
        except Exception:
            logger.warning("Could not load session latest from S3 for chat_id=%s", chat_id)

    image_token = _current_image_b64.set(latest_image)
    chat_id_token = _current_chat_id.set(chat_id)
    store_token = _result_store.set(result_store)
    try:
        start = time.time()
        agent_result = await run_agent(lc_messages)
        elapsed = round(time.time() - start, 2)
        processed_key = result_store.get("processed_image_key")
        annotated_key = result_store.get("annotated_image_key")
        display_key = processed_key or annotated_key
        annotated_url = generate_presigned_url(display_key) if display_key else None
        return ChatResponse(
            response=agent_result["response"],
            prediction_id=result_store.get("prediction_uid"),
            annotated_image_url=annotated_url,
            agent_loop_time_s=elapsed,
            iterations=agent_result["iterations"],
            tools_called=agent_result["tools_called"],
            context_limit_exceeded=agent_result["context_limit_exceeded"],
            tokens_used=agent_result["tokens_used"],
        )
    finally:
        _current_image_b64.reset(image_token)
        _current_chat_id.reset(chat_id_token)
        _result_store.reset(store_token)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
