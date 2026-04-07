from fastapi.testclient import TestClient
import sys
import os
import mlflow

# Point to the correct mlflow.db before importing app
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
mlflow.set_tracking_uri("sqlite:///mlflow.db")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.main import app

client = TestClient(app)

sample = {
    "tenure": 12, "MonthlyCharges": 65.5, "TotalCharges": 786.0,
    "gender": 0, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
    "PhoneService": 1, "MultipleLines": 0, "InternetService": 1,
    "OnlineSecurity": 0, "OnlineBackup": 1, "DeviceProtection": 0,
    "TechSupport": 0, "StreamingTV": 1, "StreamingMovies": 0,
    "Contract": 0, "PaperlessBilling": 1, "PaymentMethod": 2
}

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "running" in r.json()["message"]

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_predict():
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    data = r.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_level" in data
    assert 0 <= data["churn_probability"] <= 1

def test_batch_predict():
    r = client.post("/batch_predict", json={"customers": [sample, sample]})
    assert r.status_code == 200
    data = r.json()
    assert data["total_customers"] == 2
    assert len(data["predictions"]) == 2