import os
import unittest
from fastapi.testclient import TestClient
from app import app

TEST_IMAGE = os.path.join(os.path.dirname(__file__), "data", "beatles.jpeg")


class TestPredictionTime(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

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
