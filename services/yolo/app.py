from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import FileResponse, Response
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ultralytics import YOLO
from PIL import Image
import logging
import os
import uuid
import shutil
import time
import signal
import sys

from db import get_db, engine
from models import Base, PredictionSession, DetectionObjectModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Disable GPU usage
import torch
torch.cuda.is_available = lambda: False

app = FastAPI()

# Create tables at module level so they exist on import (required for pytest)
Base.metadata.create_all(bind=engine)

# Expose /metrics endpoint with default process metrics + FastAPI HTTP metrics
Instrumentator().instrument(app).expose(app)

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

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

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


@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Predict objects in an image
    """
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only image files are supported")

    start_time = time.time()

    uid = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)
    original_path = os.path.join(UPLOAD_DIR, uid + ext)
    predicted_path = os.path.join(PREDICTED_DIR, uid + ext)

    with open(original_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = model(original_path, device="cpu", conf=CONFIDENCE_THRESHOLD)

    annotated_frame = results[0].plot()
    annotated_image = Image.fromarray(annotated_frame)
    annotated_image.save(predicted_path)

    session = PredictionSession(uid=uid, original_image=original_path, predicted_image=predicted_path)
    db.add(session)
    db.flush()  # flush so FK is satisfied before adding objects

    detection_objects = []
    for idx, box in enumerate(results[0].boxes):
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        obj = DetectionObjectModel(prediction_uid=uid, label=label, score=score, box=str(bbox))
        db.add(obj)
        detection_objects.append(DetectionObject(id=idx, label=label, score=score, box=bbox))

    db.commit()

    return PredictResponse(
        uid=uid,
        timestamp=timestamp,
        original_image=original_path,
        predicted_image=predicted_path,
        detection_objects=detection_objects,
        processing_time_s=round(time.time() - start_time, 2),
    )

@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str, db: Session = Depends(get_db)):
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
            } for obj in objects
        ]
    }



@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str, db: Session = Depends(get_db)):
    """
    Return the annotated (bounding-box) image for a prediction
    """
    session = db.query(PredictionSession).filter(PredictionSession.uid == uid).first()
    if not session or not os.path.exists(session.predicted_image):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(session.predicted_image)

# when getting get request with empty label we return 400 error
# this is necessary because this endpoint doesnt match the endpoint with label parameter, so we need to handle it separately
@app.get("/predictions/label/")
def get_predictions_by_empty_label():
    """
    Return 400 when label is empty
    """
    raise HTTPException(status_code=400, detail="Label cannot be empty")



@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str, db: Session = Depends(get_db)):
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


# when getting get request we run this function and we return all the objects that have a score greater than or equal to the min_score
@app.get("/predictions/score/{min_score}")
def get_predictions_by_score(min_score: float, db: Session = Depends(get_db)):
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
    #  uvicorn is a server for running fastapi applications.
    import uvicorn

    def handle_sigterm(signum, frame):
        global is_shutting_down
        is_shutting_down = True
        logging.info("Received SIGTERM. Shutting down gracefully...")
        logging.info("Cleanup done. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    uvicorn.run(app, host="0.0.0.0", port=8080)
