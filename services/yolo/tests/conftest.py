import io
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.5")

import app as app_module
from app import app
from db import get_db
from models import Base, PredictionSession, DetectionObjectModel


# ── Fake YOLO model ───────────────────────────────────────────────────────────

class FakeValue:
    def __init__(self, v):
        self.value = v

    def item(self):
        return self.value


class FakeBoxCoordinates:
    def tolist(self):
        return [10, 20, 100, 200]


class FakeBox:
    cls = [FakeValue(0)]
    conf = [0.91]
    xyxy = [FakeBoxCoordinates()]


class FakeResult:
    boxes = [FakeBox()]

    def plot(self):
        return np.zeros((20, 20, 3), dtype=np.uint8)


class FakeModel:
    names = {0: "person"}

    def __call__(self, path, device="cpu", conf=0.5):
        return [FakeResult()]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def setup_db_and_dirs(tmp_path, monkeypatch):
    test_engine = create_engine(
        f"sqlite:///{tmp_path}/test.db", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(tmp_path / "uploads/original"))
    monkeypatch.setattr(app_module, "PREDICTED_DIR", str(tmp_path / "uploads/predicted"))
    os.makedirs(str(tmp_path / "uploads/original"), exist_ok=True)
    os.makedirs(str(tmp_path / "uploads/predicted"), exist_ok=True)
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def seeded_db():
    """
    Insert two prediction sessions with three detected objects into the DB.

    Session abc-123: person (score 0.91) + car (score 0.50)
    Session def-456: dog (score 0.49)
    """
    db: Session = next(app.dependency_overrides[get_db]())
    db.add_all([
        PredictionSession(uid="abc-123", original_image="uploads/original/abc-123.jpg",
                          predicted_image="uploads/predicted/abc-123.jpg"),
        PredictionSession(uid="def-456", original_image="uploads/original/def-456.jpg",
                          predicted_image="uploads/predicted/def-456.jpg"),
    ])
    db.flush()
    db.add_all([
        DetectionObjectModel(prediction_uid="abc-123", label="person", score=0.91, box="[10, 20, 100, 200]"),
        DetectionObjectModel(prediction_uid="abc-123", label="car", score=0.50, box="[1, 2, 3, 4]"),
        DetectionObjectModel(prediction_uid="def-456", label="dog", score=0.49, box="[5, 6, 7, 8]"),
    ])
    db.commit()
    db.close()


# ── ORM seed helpers (used by test files instead of old sqlite3 helpers) ──────

def save_prediction_session(uid, original_image, predicted_image):
    db: Session = next(app.dependency_overrides[get_db]())
    db.add(PredictionSession(uid=uid, original_image=original_image, predicted_image=predicted_image))
    db.commit()
    db.close()


def save_detection_object(prediction_uid, label, score, box):
    db: Session = next(app.dependency_overrides[get_db]())
    db.add(DetectionObjectModel(prediction_uid=prediction_uid, label=label, score=score, box=str(box)))
    db.commit()
    db.close()


def make_jpeg_bytes():
    """Return a minimal in-memory JPEG for upload tests."""
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), color="white").save(buf, format="JPEG")
    buf.seek(0)
    return buf
