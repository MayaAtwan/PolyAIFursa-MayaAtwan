import os
import tempfile
import unittest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import app
from db import get_db
from models import Base

TEST_IMAGE = os.path.join(os.path.dirname(__file__), "data", "beatles.jpeg")


class TestPredictionTime(unittest.TestCase):
    def setUp(self):
        _, self._db_file = tempfile.mkstemp(suffix=".db")
        test_engine = create_engine(
            f"sqlite:///{self._db_file}", connect_args={"check_same_thread": False}
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
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.pop(get_db, None)
        os.unlink(self._db_file)

    def test_predict_includes_processing_time(self):
        with open(TEST_IMAGE, "rb") as f:
            response = self.client.post(
                "/predict",
                files={"file": ("beatles.jpeg", f, "image/jpeg")}
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("processing_time_s", data)
        self.assertIsInstance(data["processing_time_s"], (int, float))
        self.assertGreaterEqual(data["processing_time_s"], 0)
