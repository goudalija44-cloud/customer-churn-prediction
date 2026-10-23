
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best_cv_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predict customer churn probability using a trained "
        "Gradient Boosting model."
    ),
    version="1.0.0",
)


class CustomerData(BaseModel):
    customerID: str = Field(..., min_length=1)
    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0, le=1000)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        # Convert the request to a DataFrame using the model's input fields.
        customer_dict = customer.model_dump()
        customer_dict.pop("customerID")

        input_df = pd.DataFrame([customer_dict])

        # Use the same 0.34 decision threshold evaluated earlier.
        probabilities = model.predict_proba(input_df)[0]
        churn_probability = float(probabilities[1])
        threshold = 0.34

        prediction = (
            "Churn" if churn_probability >= threshold else "No Churn"
        )

        return {
            "customerID": customer.customerID,
            "prediction": prediction,
            "churn_probability": round(churn_probability, 4),
            "churn_probability_percent": round(
                churn_probability * 100, 2
            ),
            "threshold": threshold,
            "model": "Gradient Boosting",
        }

    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Unable to process customer data: {exc}",
        ) from exc
