import os
import mlflow.sklearn
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from api.schemas import (
    CustomerFeatures, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="MLOps pipeline for predicting customer churn using Random Forest",
    version="1.0.0"
)

MODEL_URI = "models:/churn-model/latest"
FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

try:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.sklearn.load_model(MODEL_URI)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None


def make_prediction(customer: CustomerFeatures) -> PredictionResponse:
    df = pd.DataFrame([customer.model_dump()])[FEATURE_ORDER]
    prob = model.predict_proba(df)[0][1]
    prediction = prob >= 0.5
    risk = "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"
    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=prediction,
        risk_level=risk
    )


@app.get("/")
def root():
    return {"message": "Churn Prediction API is running!", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return make_prediction(customer)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        preds = [make_prediction(c) for c in request.customers]
        high_risk = sum(1 for p in preds if p.risk_level == "High")
        return BatchPredictionResponse(
            predictions=preds,
            total_customers=len(preds),
            high_risk_count=high_risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))