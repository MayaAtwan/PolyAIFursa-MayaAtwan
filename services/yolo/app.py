from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, Response
from prometheus_fastapi_instrumentator import Instrumentator
from ultralytics import YOLO
from PIL import Image
import sqlite3
import logging
import os
import uuid
import shutil
import time
import signal
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Disable GPU usage
import torch
torch.cuda.is_available = lambda: False

app = FastAPI()

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
DB_PATH = "predictions.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")  

# Initialize SQLite
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        # Create the predictions main table to store the prediction session
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_sessions (
                uid TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                original_image TEXT,
                predicted_image TEXT
            )
        """)
        
        # Create the objects table to store individual detected objects in a given image
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_uid TEXT,
                label TEXT,
                score REAL,
                box TEXT,
                FOREIGN KEY (prediction_uid) REFERENCES prediction_sessions (uid)
            )
        """)
        
        # Create index for faster queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_uid ON detection_objects (prediction_uid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON detection_objects (label)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_score ON detection_objects (score)")


# helper functopn to insert prediction session into the database, we use this function to save the original and predicted image paths for each prediction session, along with a unique identifier (uid) that can be used to retrieve the session later. The function takes three parameters: uid (a unique identifier for the prediction session), original_image (the file path of the original uploaded image), and predicted_image (the file path of the annotated image with detected objects). It establishes a connection to the SQLite database, executes an INSERT statement to add a new record to the prediction_sessions table, and then closes the connection automatically when done.
def save_prediction_session(uid, original_image, predicted_image):
    """
    Save prediction session to database
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO prediction_sessions (uid, original_image, predicted_image)
            VALUES (?, ?, ?)
        """, (uid, original_image, predicted_image))
# helper function to insert detected objects into the database, we use this function to save the details of each detected object for a given prediction session. The function takes four parameters: prediction_uid (the unique identifier of the prediction session that this object belongs to), label (the class label of the detected object), score (the confidence score of the detection), and box (the bounding box coordinates of the detected object). It establishes a connection to the SQLite database, executes an INSERT statement to add a new record to the detection_objects table with the provided details, and then closes the connection automatically when done.
def save_detection_object(prediction_uid, label, score, box):
    """
    Save detection object to database
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO detection_objects (prediction_uid, label, score, box)
            VALUES (?, ?, ?, ?)
        """, (prediction_uid, label, score, str(box)))


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """
    Predict objects in an image
    """
    # Validate file type
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only image files are supported")
    
    start_time = time.time()

    # Generate a unique ID for this prediction session and save the original and predicted images
    uid = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_DIR, uid + ext)
    predicted_path = os.path.join(PREDICTED_DIR, uid + ext)

    # Save the uploaded file to disk, wb mode to handle binary data correctly
    # wb means "write binary" and is necessary for saving image files without corruption
    with open(original_path, "wb") as f:
        shutil.copyfileobj(file.file, f) #shutil means to copy file-like objects efficiently without loading the entire file into memory

    # Run the YOLO model on the saved image with the specified confidence threshold
    results = model(original_path, device="cpu", conf=CONFIDENCE_THRESHOLD)

    # Save the annotated image with bounding boxes drawn on it
    annotated_frame = results[0].plot()  # NumPy image with boxes
    annotated_image = Image.fromarray(annotated_frame)
    annotated_image.save(predicted_path)

    # Save the prediction session to the database
    save_prediction_session(uid, original_path, predicted_path)
    
    # Save each detected object to the database and collect labels for the response
    detected_labels = []
    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        save_detection_object(uid, label, score, bbox)
        detected_labels.append(label)

    processing_time = round(time.time() - start_time, 2)

    return {
        "prediction_uid": uid,
        "detection_count": len(results[0].boxes),
        "labels": detected_labels,
        "time_took": processing_time
    }

@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str):
    """
    Get prediction session by uid with all detected objects
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        # Get prediction session
        session = conn.execute("SELECT * FROM prediction_sessions WHERE uid = ?", (uid,)).fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Prediction not found")
            
        # Get all detection objects for this prediction
        objects = conn.execute(
            "SELECT * FROM detection_objects WHERE prediction_uid = ?", 
            (uid,)
        ).fetchall()
        
        return {
            "uid": session["uid"],
            "timestamp": session["timestamp"],
            "original_image": session["original_image"],
            "predicted_image": session["predicted_image"],
            "detection_objects": [
                {
                    "id": obj["id"],
                    "label": obj["label"],
                    "score": obj["score"],
                    "box": obj["box"]
                } for obj in objects
            ]
        }



@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str):
    """
    Return the annotated (bounding-box) image for a prediction
    """
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT predicted_image FROM prediction_sessions WHERE uid = ?", (uid,)
        ).fetchone()
    if not row or not os.path.exists(row[0]):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(row[0])

# when getting get request with empty label we return 400 error
# this is necessary beacuse this endpoint doesnt match the endpoint with label parameter, so we need to handle it separately
@app.get("/predictions/label/")
def get_predictions_by_empty_label():
    """
    Return 400 when label is empty
    """
    raise HTTPException(status_code=400, detail="Label cannot be empty")



@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str):
    """
    Return all prediction sessions that contain at least one detected object
    with the given label
    """
    if label.strip() == "": # if the label is empty or only contains whitespace, we raise a 400 error to indicate that the label cannot be empty
        raise HTTPException(status_code=400, detail="Label cannot be empty")

    # Query the database for prediction sessions and their detected objects with the given label
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row # this allows us to access columns by name instead of index, making the code more readable and less error-prone when fetching data from the database

        # run the SQL query to select all prediction sessions and their detected objects that have the given label. We join the prediction_sessions table with the detection_objects table on the prediction_uid, and filter by the label. The results are ordered by timestamp and object id for consistent output.
        rows = conn.execute( 
            """
            SELECT
                prediction_sessions.uid,
                prediction_sessions.timestamp,
                detection_objects.id,
                detection_objects.label,
                detection_objects.score,
                detection_objects.box
            FROM prediction_sessions
            JOIN detection_objects
                ON prediction_sessions.uid = detection_objects.prediction_uid
            WHERE detection_objects.label = ?
            ORDER BY prediction_sessions.timestamp, detection_objects.id
            """,
            (label,),
        ).fetchall()

    sessions = {}

    for row in rows:
        uid = row["uid"]
        # if we haven't seen this uid before, we create a new entry in the sessions dictionary with the uid, timestamp, and an empty list for detection_objects. If we have seen this uid before, we simply append the current detected object to the existing list of detection_objects for that session. This way, we group all detected objects by their prediction session (uid) and return a structured response that includes the session details along with all relevant detected objects.
        if uid not in sessions:
            sessions[uid] = {
                "uid": row["uid"],
                "timestamp": row["timestamp"],
                "detection_objects": [],
            }
        # we append the current detected object (with its id, label, score, and box) to the list of detection_objects for the corresponding session (uid) in the sessions dictionary. This allows us to group all detected objects by their prediction session and return a structured response that includes the session details along with all relevant detected objects.
        sessions[uid]["detection_objects"].append(
            {
                "id": row["id"],
                "label": row["label"],
                "score": row["score"],
                "box": row["box"],
            }
        )

    return list(sessions.values())


# when getting get request we run this function and we return all the objects that have a score greater than or equal to the min_score
@app.get("/predictions/score/{min_score}") # we use the url path
def get_predictions_by_score(min_score: float): #  if the user sends string instead of float it will return 422 Unprocessable Entity error because FastAPI will try to convert the path parameter to float and fail if it's not a valid float
    """
    Return all detection objects with confidence score greater than
    or equal to min_score
    """
    if not 0.0 <= min_score <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="min_score must be between 0.0 and 1.0",
        )

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row # this allows us to access columns by name instead of index, making the code more readable and less error-prone when fetching data from the database

        objects = conn.execute(
            """
            SELECT id, prediction_uid, label, score, box
            FROM detection_objects
            WHERE score >= ?
            ORDER BY id
            """,
            (min_score,),
        ).fetchall() # we execute the SQL query to select all detection objects with a score greater than or equal to min_score, and we order the results by id for consistent output. The results are fetched as a list of rows

    return [
        { # we return a list of dictionaries, where each dictionary represents a detection object with its id, prediction_uid, label, score, and box coordinates
            "id": obj["id"],
            "prediction_uid": obj["prediction_uid"],
            "label": obj["label"],
            "score": obj["score"],
            "box": obj["box"],
        }
        for obj in objects
    ]

is_shutting_down = False
@app.get("/health2")
def health():
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

    init_db()

    uvicorn.run(app, host="0.0.0.0", port=8080)
