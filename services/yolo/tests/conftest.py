import io
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.5")

import app as app_module
from app import app, init_db, save_detection_object, save_prediction_session


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
    upload_dir = str(tmp_path / "uploads" / "original")
    predicted_dir = str(tmp_path / "uploads" / "predicted")

    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(predicted_dir, exist_ok=True)

    monkeypatch.setattr(app_module, "DB_PATH", db_file)
    monkeypatch.setattr(app_module, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(app_module, "PREDICTED_DIR", predicted_dir)

    init_db()


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
    save_prediction_session(
        "abc-123",
        "uploads/original/abc-123.jpg",
        "uploads/predicted/abc-123.jpg",
    )
    save_detection_object("abc-123", "person", 0.91, [10, 20, 100, 200])
    save_detection_object("abc-123", "car", 0.50, [1, 2, 3, 4])

    save_prediction_session(
        "def-456",
        "uploads/original/def-456.jpg",
        "uploads/predicted/def-456.jpg",
    )
    save_detection_object("def-456", "dog", 0.49, [5, 6, 7, 8])


def make_jpeg_bytes():
    """Return a minimal in-memory JPEG for upload tests."""
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), color="white").save(buf, format="JPEG")
    buf.seek(0)
    return buf
