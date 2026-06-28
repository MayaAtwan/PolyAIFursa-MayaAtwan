# Vision Agent

A LangChain-powered AI vision agent with a manual ReAct loop. Accepts text and base64-encoded images, and can call tools (e.g. YOLO object detection) to answer questions.

## Image flow (S3)

The agent and YOLO service are decoupled via Amazon S3 — the image bytes are **not**
sent over HTTP between them:

1. The user sends an image (base64) to `POST /chat`.
2. The agent uploads the original image to S3 under
   `<chat_id>/<prediction_id>/original/image.jpg`.
3. The agent calls the YOLO service with **only the S3 object key**
   (`{"image_s3_key": "<key>"}`), not the bytes.
4. YOLO downloads the original from S3, runs detection, uploads the annotated
   image back to S3 under the matching `<chat_id>/<prediction_id>/predicted/...`
   key, and returns JSON.

`chat_id` may be supplied by the client in the request body; if absent the agent
generates one per request.

## Prerequisites

- Python 3.10+
- A running YOLO service (optional - only needed for `detect_objects`)


## Setup

Install dependencies (from `services/agent/`):

```bash
pip install -r requirements.txt
```

Configure environment:

```bash
cp .env.example .env
# Edit .env and set at least OPENAI_API_KEY (or another provider key) and MODEL
```

`.env` variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | - | Required for OpenAI models |
| `ANTHROPIC_API_KEY` | - | Required for Anthropic models |
| `GOOGLE_API_KEY` | - | Required for Google models |
| `AWS_REGION` | `us-east-1` | Region for AWS Bedrock models and the S3 bucket |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | - | AWS credentials for Bedrock + S3 (or use a shared profile / instance role) |
| `AWS_S3_BUCKET` | - | S3 bucket used to hand images to YOLO (e.g. `maya-polyai-images`) |
| `MODEL` | `claude-sonnet-4-6` | Any model string supported by `init_chat_model` |
| `YOLO_SERVICE_URL` | `http://localhost:8080` | URL of the YOLO microservice |

### Using AWS Bedrock

Bedrock models are selected with the `bedrock_converse:` prefix (the Converse API supports tool
calling, which the agent's YOLO tools require), e.g.
`MODEL=bedrock_converse:anthropic.claude-3-haiku-20240307-v1:0`. The `langchain-aws` package
(in `requirements.txt`) provides the integration, and credentials are read from the standard AWS
chain (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, a shared profile, or an instance role).

Each model must be enabled for your account under **Model access** in the Bedrock console for the
chosen `AWS_REGION` before it can be invoked. Tool calling works with the Claude, Amazon Nova, and
Llama models; `mistral.mistral-7b-instruct-v0:2` has no Bedrock tool support, so the YOLO tools
won't fire on it.

## Running

```bash
cd services/agent
python app.py
```

The server starts at `http://localhost:8000`.

## Testing with curl

### Health check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok"}
```

### Plain text message

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! What can you do?"}'
```

### Send a message with an image

```bash
echo "{\"message\": \"What objects are in this image?\", \"image_base64\": \"$(base64 -w0 beatles.jpeg)\"}" \
  | curl -X POST http://localhost:8000/chat \
         -H "Content-Type: application/json" \
         -d @-
```

## API Reference

### `POST /chat`

Request body:

```json
{
  "message": "string (optional, defaults to 'What's in this image?')",
  "image_base64": "string (optional, base64-encoded JPEG or PNG)"
}
```

Response:

```json
{
  "response": "string"
}
```

### `GET /health`

Returns `{"status": "ok"}` when the service is running.
