# YOLO Object Detection Service

This is a FastAPI-based web service that performs object detection using the YOLOv8 model. Images are exchanged with the Agent service via Amazon S3: `POST /predict` receives an **S3 object key**, downloads the original image from S3, runs detection, uploads the annotated image back to S3, and stores prediction results in a SQLAlchemy-backed database (SQLite by default, Postgres optional) for later retrieval.

## Setup Instructions

1. Make sure the shared project virtualenv is activated (see the root README).

1. Install requirements (from `services/yolo/`):

```bash
pip install -r torch-requirements.txt
pip install -r requirements.txt
```

1. Run the application:

```bash
python app.py
```

The service will be available at http://<your_server_ip>:8080

You can test the api endpoints using `curl` or Postman. See the API Endpoints section below for details on available endpoints and how to use them.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence score (0.0–1.0) for a detection to be reported. Raise it to get only high-confidence results; lower it to catch more objects. |
| `AWS_REGION` | `us-east-1` | AWS region of the S3 bucket |
| `AWS_S3_BUCKET` | - | S3 bucket to read originals from / write predicted images to (e.g. `maya-polyai-images`) |
| `DB_BACKEND` | `sqlite` | `sqlite` (default) or `postgres`; Postgres also reads `DB_USER`/`DB_PASSWORD`/`DB_HOST`/`DB_PORT`/`DB_NAME` |

AWS credentials are read from the standard AWS credential chain
(`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, a shared profile, or an instance role).

Example:
```bash
export CONFIDENCE_THRESHOLD=0.7
python app.py
```

## Running Tests

The test suite uses `pytest` and FastAPI's built-in test client — no running server needed.

```bash
pytest tests/
```


## API Endpoints

* `POST /predict` - Run object detection on an image stored in S3 (body: `{"image_s3_key": "<key>"}`)
* `GET /prediction/{uid}` - Get details of a specific prediction by ID
* `GET /predictions/label/{label}` - Get all predictions containing a specific object label (e.g., "person", "car")
* `GET /predictions/score/{min_score}` - Get predictions with confidence score above threshold (e.g., 0.5)
* `GET /prediction/{uid}/image` - Get the processed image with detection boxes
* `GET /image/{type}/{filename}` - Get original or predicted image by filename

## Testing the API

You can use tools like curl, Postman, or a web browser to test the endpoints. For example:

1. Run a prediction on an image already uploaded to S3:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"image_s3_key": "<chat_id>/<prediction_id>/original/image.jpg"}'
```

2. View detection results (replace {uid} with the ID returned from the upload):
```bash
curl http://localhost:8080/prediction/{uid} 