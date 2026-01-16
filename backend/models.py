from pydantic import BaseModel, Field
from typing import Dict, List

class PatientData(BaseModel):
    """Patient health data for prediction."""
    age: float = Field(..., ge=20, le=80, description="Age in years")
    bmi: float = Field(..., ge=18, le=45, description="Body Mass Index")
    glucose: float = Field(..., ge=70, le=200, description="Glucose level (mg/dL)")
    blood_pressure: float = Field(..., ge=90, le=180, description="Systolic blood pressure (mmHg)")
    cholesterol: float = Field(..., ge=150, le=300, description="Cholesterol level (mg/dL)")
    smoking: int = Field(..., ge=0, le=1, description="Smoking status (0: non-smoker, 1: smoker)")
    exercise: int = Field(..., ge=0, le=5, description="Exercise days per week")
    family_history: int = Field(..., ge=0, le=1, description="Family history of disease (0: no, 1: yes)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 45,
                "bmi": 28.5,
                "glucose": 140,
                "blood_pressure": 130,
                "cholesterol": 220,
                "smoking": 0,
                "exercise": 2,
                "family_history": 1
            }
        }

class PredictionResponse(BaseModel):
    """Response model for disease risk prediction."""
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    risk_code: int = Field(..., description="Risk code: 0 (Low), 1 (Medium), 2 (High)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probability for each risk level")
    model_type: str = Field(..., description="Model used for prediction")

class ExplanationResponse(BaseModel):
    """Response model for XAI explanation."""
    prediction: PredictionResponse
    shap_explanation: Dict = Field(..., description="SHAP feature importance")
    lime_explanation: Dict = Field(..., description="LIME explanation with text")
    feature_importance_chart: str = Field(..., description="Base64 encoded SHAP visualization")

class ModelComparisonResponse(BaseModel):
    """Response model for comparing all models."""
    patient_data: PatientData
    predictions: Dict[str, PredictionResponse]
    
class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    available_models: List[str]
