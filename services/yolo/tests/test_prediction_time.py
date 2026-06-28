import unittest

from fastapi.testclient import TestClient
from app import app


class TestPredictionTime(unittest.TestCase):
    # The autouse `setup_db_and_dirs` fixture in conftest.py wires up an isolated
    # DB and the fake S3 helpers around every test, including these unittest cases.
    def setUp(self):
        self.client = TestClient(app)

    def test_predict_includes_processing_time(self):
        response = self.client.post(
            "/predict",
            json={"image_s3_key": "chat-t/pred-t/original/beatles.jpeg"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("processing_time_s", data)
        self.assertIsInstance(data["processing_time_s"], (int, float))
        self.assertGreaterEqual(data["processing_time_s"], 0)
