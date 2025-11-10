# app/api_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import numpy as np
from src.inference import predict_from_list

app = FastAPI(title="Predictive Maintenance API")

class FeaturesIn(BaseModel):
    features: list  # list of numbers, same order as _ordered_feature_names()

@app.on_event("startup")
def load_models():
    # lazy-load done inside predict_from_list; keep this hook for extension
    pass

@app.post("/predict")
def predict(payload: FeaturesIn):
    try:
        features = payload.features
        pred, prob = predict_from_list(features)
        return {"prediction": int(pred), "probability": float(prob) if prob is not None else None}
    except Exception as e:
        return {"error": str(e)}
