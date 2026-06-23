# pytest has a special rule: any file named conftest.py is automatically imported by pytest
# and loaded before any test runs.

import io  # this is used to create an in-memory and not on-disk file object for testing image uploads
import os

import numpy as np
import pytest
# without running a live server, we can use the TestClient to make requests to the API endpoints and test their behavior
from fastapi.testclient import TestClient  # this is used to create a test client for the fastapi application
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from PIL import Image

os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.5")

from app import app
import app as app_module
from db import get_db
from models import Base, PredictionSession, DetectionObjectModel


# ── Fake YOLO model ───────────────────────────────────────────────────────────
# Replaces the real model so tests never download weights or run inference.

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
    # call method allows an instance of the class to be called as a function, so when we call the fake model with an image path, it returns a list containing a single FakeResult object
    def __call__(self, path, device="cpu", conf=0.5):
        return [FakeResult()]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def setup_db_and_dirs(tmp_path, monkeypatch):
    """
    SETUP:    Create an isolated temp DB and upload dirs; patch module globals.
    TEARDOWN: pytest removes tmp_path automatically after each test — no manual
              cleanup needed.
    """
    db_file = str(tmp_path / "test.db")
    test_engine = create_engine(
        f"sqlite:///{db_file}", connect_args={"check_same_thread": False}
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

    upload_dir = str(tmp_path / "uploads" / "original")
    predicted_dir = str(tmp_path / "uploads" / "predicted")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(predicted_dir, exist_ok=True)
    monkeypatch.setattr(app_module, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(app_module, "PREDICTED_DIR", predicted_dir)

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
        DetectionObjectModel(prediction_uid="abc-123", label="car",    score=0.50, box="[1, 2, 3, 4]"),
        DetectionObjectModel(prediction_uid="def-456", label="dog",    score=0.49, box="[5, 6, 7, 8]"),
    ])
    db.commit()
    db.close()


# Creates an in-memory buffer — think of it as a fake file that lives in RAM instead of on your hard drive.
# It behaves exactly like a real file (you can read and write to it), but nothing touches the disk.
def make_jpeg_bytes():
    """Return a minimal in-memory JPEG for upload tests."""
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), color="white").save(buf, format="JPEG")
    buf.seek(0)
    return buf
