from typing import Dict

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    disease: str
    features: Dict[str, int | float]


# Backward-compatible alias for older imports
PedictionRequestLegacy = PredictionRequest
PedictionRequestAlias = PredictionRequest
PedictionRequestTypo = PredictionRequest


class PredictionResponse(BaseModel):
    disease: str
    prediction: int
    probability: float
    

    