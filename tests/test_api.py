from fastapi.testclient import TestClient
from src.api import app, ml_models
from PIL import Image
import io
from unittest.mock import patch
import pytest

# in order to avoid loading real models, a mock service is a better choice
class FakeInference:
    def prepare_image(self, image):
        return(image, 0, 0)
    def predict(self, image):
        return ((image, 100), "dummy_prediction")

client = TestClient(app)    

def inject_fake_model():
    ml_models["inference"] = FakeInference()

    yield

    ml_models.clear()

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

def test_invalid_file_format():
    with TestClient(app) as client:
        dummy_file = {"file": ("test.txt", b"This is not an image data", "text/plain")}
        response = client.post ("/predict", files = dummy_file)
        assert response.status_code == 415
        assert response.json()["detail"] == "File format should be either jpeg or png"

def test_no_file_uploaded():
    with TestClient(app) as client:
        response = client.post("/predict")
        assert response.status_code == 422

def test_predict_success():
    dummy_image = Image.new("RGB", (100, 100), color="white")
    virtual_file = io.BytesIO()
    dummy_image.save(virtual_file, format = "JPEG")
    virtual_file.seek(0) # return to the beginning of the file inside RAM
    valid_file = {"file": ("test.jpg", virtual_file.read(), "image/jpeg")}
    with TestClient(app) as client:
        response = client.post("/predict", files = valid_file)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert len(response.content) > 0



