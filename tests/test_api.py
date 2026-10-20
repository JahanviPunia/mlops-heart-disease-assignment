import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from api import app

client = TestClient(app)

def sample_patient():
    return {
        "age": 55,
        "sex": 1,
        "cp": 0,
        "trestbps": 130,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True

def test_prediction_endpoint():
    response = client.post("/predict", json=sample_patient())
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in [0, 1]
    assert 0 <= body["positive_class_probability"] <= 1
    assert body["model"] == "Logistic Regression"
    assert "disclaimer" in body

def test_rejects_unexpected_input():
    patient = sample_patient()
    patient["unexpected_field"] = 123

    response = client.post("/predict", json=patient)
    assert response.status_code == 422

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.json()
