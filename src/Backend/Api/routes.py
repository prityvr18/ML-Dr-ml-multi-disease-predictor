from fastapi import APIRouter

from src.Backend.schemas.prediction_schema import (
    PredictionRequest,
    PredictionResponse,
)
from src.Backend.services.predictor import predict_disease

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "API is healthy and running",
    }


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Endpoint to predict the disease based on input features.
    """
    disease = request.disease
    features = request.features
    result = predict_disease(disease=disease, input_data=features)
    return PredictionResponse(**result)
