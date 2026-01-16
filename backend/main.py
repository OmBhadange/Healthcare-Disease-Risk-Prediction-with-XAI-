from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import joblib
from models import (
    PatientData, 
    PredictionResponse, 
    ExplanationResponse,
    ModelComparisonResponse,
    HealthCheckResponse
)
from xai_explainer import XAIExplainer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare Disease Risk Prediction API",
    description="ML/DL-based disease risk prediction with Explainable AI (SHAP & LIME)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
scaler = None
lr_model = None
rf_model = None
nn_model = None
xai_explainer = None

@app.on_event("startup")
async def load_models():
    """Load all trained models on startup."""
    global scaler, lr_model, rf_model, nn_model, xai_explainer
    
    try:
        logger.info("Loading models...")
        scaler = joblib.load('models/scaler.pkl')
        lr_model = joblib.load('models/logistic_regression.pkl')
        rf_model = joblib.load('models/random_forest.pkl')
        nn_model = joblib.load('models/neural_network.pkl')
        xai_explainer = XAIExplainer()
        logger.info("All models loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Root endpoint - API health check."""
    return {
        "status": "healthy",
        "models_loaded": all([scaler, lr_model, rf_model, nn_model]),
        "available_models": ["logistic_regression", "random_forest", "neural_network"]
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": all([scaler, lr_model, rf_model, nn_model]),
        "available_models": ["logistic_regression", "random_forest", "neural_network"]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(
    patient_data: PatientData,
    model_type: str = "random_forest"
):
    """
    Predict disease risk for a patient.
    
    Args:
        patient_data: Patient health information
        model_type: Model to use (logistic_regression, random_forest, neural_network)
    
    Returns:
        Risk prediction with confidence score
    """
    try:
        # Validate model type
        if model_type not in ["logistic_regression", "random_forest", "neural_network"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid model_type. Choose: logistic_regression, random_forest, or neural_network"
            )
        
        # Prepare input
        patient_dict = patient_data.model_dump()
        X_input = _prepare_input(patient_dict)
        
        # Select model and predict
        if model_type == "logistic_regression":
            model = lr_model
        elif model_type == "random_forest":
            model = rf_model
        else:  # neural_network
            model = nn_model
        
        # Get prediction
        if model_type == "neural_network":
            prediction = int(model.predict(X_input)[0])
            prediction_proba = model.predict_proba(X_input)[0]
        else:
            prediction = int(model.predict(X_input)[0])
            prediction_proba = model.predict_proba(X_input)[0]
        
        # Format response
        risk_levels = {0: "Low", 1: "Medium", 2: "High"}
        
        return {
            "risk_level": risk_levels[prediction],
            "risk_code": prediction,
            "confidence": float(prediction_proba[prediction]),
            "probabilities": {
                "Low": float(prediction_proba[0]),
                "Medium": float(prediction_proba[1]),
                "High": float(prediction_proba[2])
            },
            "model_type": model_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(
    patient_data: PatientData,
    model_type: str = "random_forest"
):
    """
    Get XAI explanation for a prediction using SHAP and LIME.
    
    Args:
        patient_data: Patient health information
        model_type: Model to use for explanation
    
    Returns:
        Prediction with SHAP and LIME explanations
    """
    try:
        # Validate model type
        if model_type not in ["logistic_regression", "random_forest", "neural_network"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid model_type. Choose: logistic_regression, random_forest, or neural_network"
            )
        
        # Get patient data as dict
        patient_dict = patient_data.model_dump()
        
        # Get combined explanation
        explanation = xai_explainer.get_combined_explanation(patient_dict, model_type)
        
        # Get prediction
        shap_data = explanation['shap']
        lime_data = explanation['lime']
        
        # Generate visualization
        chart_base64 = xai_explainer.visualize_shap_waterfall(patient_dict, model_type)
        
        # Format prediction response
        risk_levels = {0: "Low", 1: "Medium", 2: "High"}
        prediction_response = {
            "risk_level": risk_levels[shap_data['prediction']],
            "risk_code": shap_data['prediction'],
            "confidence": shap_data['confidence'],
            "probabilities": {
                "Low": shap_data['prediction_proba'][0],
                "Medium": shap_data['prediction_proba'][1],
                "High": shap_data['prediction_proba'][2]
            },
            "model_type": model_type
        }
        
        return {
            "prediction": prediction_response,
            "shap_explanation": {
                "feature_importance": shap_data['feature_importance']
            },
            "lime_explanation": {
                "feature_contributions": lime_data['feature_contributions'],
                "explanation_text": lime_data['explanation_text']
            },
            "feature_importance_chart": chart_base64
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.post("/models/compare", response_model=ModelComparisonResponse)
async def compare_models(patient_data: PatientData):
    """
    Compare predictions from all three models.
    
    Args:
        patient_data: Patient health information
    
    Returns:
        Predictions from all models
    """
    try:
        predictions = {}
        
        for model_type in ["logistic_regression", "random_forest", "neural_network"]:
            # Get prediction
            patient_dict = patient_data.model_dump()
            X_input = _prepare_input(patient_dict)
            
            if model_type == "logistic_regression":
                model = lr_model
            elif model_type == "random_forest":
                model = rf_model
            else:
                model = nn_model
            
            # Predict
            prediction = int(model.predict(X_input)[0])
            prediction_proba = model.predict_proba(X_input)[0]
            
            # Format
            risk_levels = {0: "Low", 1: "Medium", 2: "High"}
            predictions[model_type] = {
                "risk_level": risk_levels[prediction],
                "risk_code": prediction,
                "confidence": float(prediction_proba[prediction]),
                "probabilities": {
                    "Low": float(prediction_proba[0]),
                    "Medium": float(prediction_proba[1]),
                    "High": float(prediction_proba[2])
                },
                "model_type": model_type
            }
        
        return {
            "patient_data": patient_data,
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Model comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

def _prepare_input(patient_dict):
    """Prepare patient data for model input."""
    import pandas as pd
    
    feature_names = [
        'age', 'bmi', 'glucose', 'blood_pressure',
        'cholesterol', 'smoking', 'exercise', 'family_history'
    ]
    
    df = pd.DataFrame([patient_dict], columns=feature_names)
    X_scaled = scaler.transform(df)
    
    return X_scaled

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
