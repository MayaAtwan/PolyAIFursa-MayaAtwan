from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import Response, JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session
from ultralytics import YOLO
from PIL import Image
import logging
import os
import uuid
import time
import signal
import sys

from db import get_db, engine
from models import Base, PredictionSession, DetectionObjectModel
from s3_utils import AWS_S3_BUCKET, download_bytes, upload_bytes

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Disable GPU usage
import torch
torch.cuda.is_available = lambda: False

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "You're sending requests too quickly. Please wait a moment and try again."},
    )

# Expose /metrics endpoint with default process metrics + FastAPI HTTP metrics
Instrumentator().instrument(app).expose(app)

# Create tables at import time so pytest (which imports this module) has them ready.
Base.metadata.create_all(bind=engine)

if not AWS_S3_BUCKET:
    logging.warning(
        "AWS_S3_BUCKET is not set; image download/upload to S3 will fail. "
        "Set AWS_S3_BUCKET (and AWS_REGION) in the environment."
    )

# Confidence threshold for object detection (0.0 - 1.0).
# Detections below this score are discarded.
# Override with: export CONFIDENCE_THRESHOLD=0.7
_raw_threshold = os.environ.get("CONFIDENCE_THRESHOLD")
if _raw_threshold is not None:
    CONFIDENCE_THRESHOLD = float(_raw_threshold)
    logging.info(f"CONFIDENCE_THRESHOLD set to {CONFIDENCE_THRESHOLD} (from environment)")
else:  # pragma: no cover
    CONFIDENCE_THRESHOLD = 0.5
    logging.info(f"CONFIDENCE_THRESHOLD not set, using default: {CONFIDENCE_THRESHOLD}")

# Local scratch dirs — YOLO works off file paths, so we stage the S3 images here.
UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")


class DetectionObject(BaseModel):
    id: int
    label: str
    score: float
    box: list[float]


class PredictResponse(BaseModel):
    uid: str
    timestamp: datetime
    original_image: str
    predicted_image: str
    detection_objects: list[DetectionObject]
    processing_time_s: float


class PredictRequest(BaseModel):
    image_s3_key: str  # e.g. "<chat_id>/<prediction_id>/original/image.jpg"


@app.post("/predict", response_model=PredictResponse)
@limiter.limit("10/minute")
def predict(request: Request, body: PredictRequest, db: Session = Depends(get_db)):
    """
    Predict objects in an image stored in S3.

    The image is identified only by its S3 object key. We download it, run YOLO,
    upload the annotated image back to S3 (mirroring the key under .../predicted/...),
    and persist both keys.
    """
    image_s3_key = body.image_s3_key
    ext = os.path.splitext(image_s3_key)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only image files are supported")

    start_time = time.time()

    uid = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)

    # Download the original image from S3 to a local scratch file (YOLO needs a path).
    original_path = os.path.join(UPLOAD_DIR, uid + ext)
    with open(original_path, "wb") as f:
        f.write(download_bytes(image_s3_key))

    results = model(original_path, device="cpu", conf=CONFIDENCE_THRESHOLD)

    annotated_frame = results[0].plot()
    annotated_image = Image.fromarray(annotated_frame)
    predicted_path = os.path.join(PREDICTED_DIR, uid + ext)
    annotated_image.save(predicted_path)

    # Mirror the key: <chat>/<pred>/original/x.jpg -> <chat>/<pred>/predicted/x.jpg
    predicted_s3_key = image_s3_key.replace("/original/", "/predicted/")
    with open(predicted_path, "rb") as f:
        upload_bytes(f.read(), predicted_s3_key, content_type=_MEDIA_TYPES.get(ext, "image/jpeg"))

    session = PredictionSession(
        uid=uid, original_image=image_s3_key, predicted_image=predicted_s3_key
    )
    db.add(session)
    db.flush()

    detection_objects = []
    for idx, box in enumerate(results[0].boxes):
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        db.add(DetectionObjectModel(prediction_uid=uid, label=label, score=score, box=str(bbox)))
        detection_objects.append(DetectionObject(id=idx, label=label, score=score, box=bbox))

    db.commit()

    return PredictResponse(
        uid=uid,
        timestamp=timestamp,
        original_image=image_s3_key,
        predicted_image=predicted_s3_key,
        detection_objects=detection_objects,
        processing_time_s=round(time.time() - start_time, 2),
    )


@app.get("/prediction/{uid}")
@limiter.limit("60/minute")
def get_prediction_by_uid(request: Request, uid: str, db: Session = Depends(get_db)):
    """
    Get prediction session by uid with all detected objects
    """
    session = db.query(PredictionSession).filter(PredictionSession.uid == uid).first()
    if not session:
        raise HTTPException(status_code=404, detail="Prediction not found")

    objects = db.query(DetectionObjectModel).filter(DetectionObjectModel.prediction_uid == uid).all()

    return {
        "uid": session.uid,
        "timestamp": session.timestamp,
        "original_image": session.original_image,
        "predicted_image": session.predicted_image,
        "detection_objects": [
            {
                "id": obj.id,
                "label": obj.label,
                "score": obj.score,
                "box": obj.box,
            }
            for obj in objects
        ],
    }


@app.get("/prediction/{uid}/image")
@limiter.limit("60/minute")
def get_prediction_image(request: Request, uid: str, db: Session = Depends(get_db)):
    """
    Return the annotated (bounding-box) image for a prediction, fetched from S3.
    """
    session = db.query(PredictionSession).filter(PredictionSession.uid == uid).first()
    if not session:
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        image_bytes = download_bytes(session.predicted_image)
    except Exception:
        raise HTTPException(status_code=404, detail="Image not found")

    ext = os.path.splitext(session.predicted_image)[1].lower()
    return Response(content=image_bytes, media_type=_MEDIA_TYPES.get(ext, "image/jpeg"))


# when getting get request with empty label we return 400 error
# this is necessary beacuse this endpoint doesnt match the endpoint with label parameter, so we need to handle it separately
@app.get("/predictions/label/")
def get_predictions_by_empty_label():
    """
    Return 400 when label is empty
    """
    raise HTTPException(status_code=400, detail="Label cannot be empty")


@app.get("/predictions/label/{label}")
@limiter.limit("60/minute")
def get_predictions_by_label(request: Request, label: str, db: Session = Depends(get_db)):
    """
    Return all prediction sessions that contain at least one detected object
    with the given label
    """
    if label.strip() == "":
        raise HTTPException(status_code=400, detail="Label cannot be empty")

    rows = (
        db.query(
            PredictionSession.uid,
            PredictionSession.timestamp,
            DetectionObjectModel.id,
            DetectionObjectModel.label,
            DetectionObjectModel.score,
            DetectionObjectModel.box,
        )
        .join(DetectionObjectModel, PredictionSession.uid == DetectionObjectModel.prediction_uid)
        .filter(DetectionObjectModel.label == label)
        .order_by(PredictionSession.timestamp, DetectionObjectModel.id)
        .all()
    )

    sessions = {}
    for row in rows:
        uid = row.uid
        if uid not in sessions:
            sessions[uid] = {
                "uid": row.uid,
                "timestamp": row.timestamp,
                "detection_objects": [],
            }
        sessions[uid]["detection_objects"].append(
            {
                "id": row.id,
                "label": row.label,
                "score": row.score,
                "box": row.box,
            }
        )

    return list(sessions.values())


@app.get("/predictions/score/{min_score}")
@limiter.limit("60/minute")
def get_predictions_by_score(request: Request, min_score: float, db: Session = Depends(get_db)):
    """
    Return all detection objects with confidence score greater than
    or equal to min_score
    """
    if not 0.0 <= min_score <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="min_score must be between 0.0 and 1.0",
        )

    objects = (
        db.query(DetectionObjectModel)
        .filter(DetectionObjectModel.score >= min_score)
        .order_by(DetectionObjectModel.id)
        .all()
    )

    return [
        {
            "id": obj.id,
            "prediction_uid": obj.prediction_uid,
            "label": obj.label,
            "score": obj.score,
            "box": obj.box,
        }
        for obj in objects
    ]


is_shutting_down = False


@app.get("/health2")
def health2():
    """
    Health check endpoint
    """
    return {"status": "ok"}


# health endpoint checks if the service is running and returns a simple JSON response with status "ok" 200.
@app.get("/health")
def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if is_shutting_down:
        raise HTTPException(status_code=503, detail="Service is shutting down")
    return {"status": "ready"}


if __name__ == "__main__":  # pragma: no cover
    #  uvicorn is a server for runing fastapi applications.
    import uvicorn

    def handle_sigterm(signum, frame):
        global is_shutting_down
        is_shutting_down = True
        logging.info("Received SIGTERM. Shutting down gracefully...")
        logging.info("Cleanup done. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    uvicorn.run(app, host="0.0.0.0", port=8080)
