from pydantic import BaseModel, Field
from typing import List


class CustomerFeatures(BaseModel):
    tenure: int = Field(..., ge=0, description="Number of months the customer has stayed")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges in USD")
    TotalCharges: float = Field(..., ge=0, description="Total charges in USD")
    gender: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    Partner: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    Dependents: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    PhoneService: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    MultipleLines: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    InternetService: int = Field(..., ge=0, le=2, description="0=DSL, 1=Fiber, 2=No")
    OnlineSecurity: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    OnlineBackup: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    DeviceProtection: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    TechSupport: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    StreamingTV: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    StreamingMovies: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    Contract: int = Field(..., ge=0, le=2, description="0=Month-to-month, 1=One year, 2=Two year")
    PaperlessBilling: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    PaymentMethod: int = Field(..., ge=0, le=3, description="0-3 encoding of payment method")

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12, "MonthlyCharges": 65.5, "TotalCharges": 786.0,
                "gender": 0, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
                "PhoneService": 1, "MultipleLines": 0, "InternetService": 1,
                "OnlineSecurity": 0, "OnlineBackup": 1, "DeviceProtection": 0,
                "TechSupport": 0, "StreamingTV": 1, "StreamingMovies": 0,
                "Contract": 0, "PaperlessBilling": 1, "PaymentMethod": 2
            }
        }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int