# pytest has a special rule: any file named conftest.py is automatically imported by
# pytest, loaded before any test runs.

import io  # in-memory (not on-disk) file object for testing image bytes
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

@pytest.fixture
def s3_store():
    """In-memory stand-in for the S3 bucket: maps object key -> bytes.
    Tests can pre-seed predicted images here for retrieval tests."""
    return {}


@pytest.fixture(autouse=True)
def setup_db_and_dirs(tmp_path, monkeypatch, s3_store):
    """
    SETUP:    Isolated temp DB (via dependency_overrides) + upload dirs, and an
              in-memory fake for the S3 helpers so no real bucket is touched.
    TEARDOWN: pytest removes tmp_path automatically; we clear dependency overrides.
    """
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

    # Fake S3: original keys download a default JPEG; everything else must be
    # explicitly seeded into s3_store, otherwise the download "fails" (404).
    def fake_download(key):
        if key in s3_store:
            return s3_store[key]
        if "/original/" in key:
            return make_jpeg_bytes().getvalue()
        raise KeyError(key)

    def fake_upload(data, key, content_type="image/jpeg"):
        s3_store[key] = data

    monkeypatch.setattr(app_module, "download_bytes", fake_download)
    monkeypatch.setattr(app_module, "upload_bytes", fake_upload)

    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


# ── ORM seed helpers (used by tests instead of the old sqlite3 helpers) ────────

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


@pytest.fixture
def seeded_db():
    """
    Insert two prediction sessions with three detected objects into the DB.

    Session abc-123: person (score 0.91) + car (score 0.50)
    Session def-456: dog (score 0.49)
    """
    save_prediction_session("abc-123", "uploads/original/abc-123.jpg", "uploads/predicted/abc-123.jpg")
    save_detection_object("abc-123", "person", 0.91, [10, 20, 100, 200])
    save_detection_object("abc-123", "car", 0.50, [1, 2, 3, 4])

    save_prediction_session("def-456", "uploads/original/def-456.jpg", "uploads/predicted/def-456.jpg")
    save_detection_object("def-456", "dog", 0.49, [5, 6, 7, 8])


def make_jpeg_bytes():
    """Return a minimal in-memory JPEG for predict tests."""
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), color="white").save(buf, format="JPEG")
    buf.seek(0)
    return buf
