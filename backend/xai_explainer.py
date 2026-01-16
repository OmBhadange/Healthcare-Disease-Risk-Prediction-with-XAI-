import numpy as np
import pandas as pd
import joblib
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt
import json
import base64
from io import BytesIO

class XAIExplainer:
    """Explainable AI module using SHAP and LIME for healthcare predictions."""
    
    def __init__(self):
        """Initialize the XAI explainer with trained models."""
        self.scaler = joblib.load('models/scaler.pkl')
        self.lr_model = joblib.load('models/logistic_regression.pkl')
        self.rf_model = joblib.load('models/random_forest.pkl')
        self.nn_model = joblib.load('models/neural_network.pkl')
        
        # Feature names
        self.feature_names = [
            'age', 'bmi', 'glucose', 'blood_pressure', 
            'cholesterol', 'smoking', 'exercise', 'family_history'
        ]
        
        # Load training data for SHAP background
        df = pd.read_csv('healthcare_dataset.csv')
        X = df.drop('disease_risk', axis=1)
        self.X_background = self.scaler.transform(X.sample(100, random_state=42))
        
    def get_shap_explanation(self, patient_data, model_type='random_forest'):
        """
        Generate SHAP explanations for a prediction.
        
        Args:
            patient_data: Dictionary with patient features
            model_type: 'logistic_regression', 'random_forest', or 'neural_network'
        
        Returns:
            Dictionary with SHAP values and feature importance
        """
        # Prepare input
        X_input = self._prepare_input(patient_data)
        
        # Select model and create explainer
        if model_type == 'logistic_regression':
            model = self.lr_model
            explainer = shap.LinearExplainer(model, self.X_background)
        elif model_type == 'random_forest':
            model = self.rf_model
            explainer = shap.TreeExplainer(model)
        else:  # neural_network
            model = self.nn_model
            # For MLPClassifier, use KernelExplainer
            explainer = shap.KernelExplainer(model.predict_proba, self.X_background)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_input)
        
        # Get prediction
        prediction = int(model.predict(X_input)[0])
        prediction_proba = model.predict_proba(X_input)[0]
        
        # Process SHAP values
        if isinstance(shap_values, list):
            # Multi-class: use values for predicted class
            shap_values_class = shap_values[prediction]
        else:
            shap_values_class = shap_values
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, feature in enumerate(self.feature_names):
            if len(shap_values_class.shape) > 1:
                importance = float(shap_values_class[0][i])
            else:
                importance = float(shap_values_class[i])
            feature_importance[feature] = importance
        
        # Sort by absolute importance
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        return {
            'prediction': int(prediction),
            'prediction_proba': prediction_proba.tolist(),
            'feature_importance': dict(sorted_features),
            'risk_level': self._get_risk_level(prediction),
            'confidence': float(prediction_proba[prediction])
        }
    
    def get_lime_explanation(self, patient_data, model_type='random_forest'):
        """
        Generate LIME explanations for a prediction.
        
        Args:
            patient_data: Dictionary with patient features
            model_type: 'logistic_regression', 'random_forest', or 'neural_network'
        
        Returns:
            Dictionary with LIME explanation
        """
        # Prepare input
        X_input = self._prepare_input(patient_data)
        
        # Select model
        if model_type == 'logistic_regression':
            model = self.lr_model
        elif model_type == 'random_forest':
            model = self.rf_model
        else:  # neural_network
            model = self.nn_model
        
        predict_fn = model.predict_proba
        
        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            self.X_background,
            feature_names=self.feature_names,
            class_names=['Low Risk', 'Medium Risk', 'High Risk'],
            mode='classification'
        )
        
        # Generate explanation
        exp = explainer.explain_instance(
            X_input[0],
            predict_fn,
            num_features=len(self.feature_names)
        )
        
        # Get prediction
        prediction_proba = predict_fn(X_input)[0]
        prediction = np.argmax(prediction_proba)
        
        # Extract feature contributions
        feature_contributions = {}
        for feature, contribution in exp.as_list():
            # Parse feature name from LIME format
            feature_name = feature.split()[0].split('<=')[0].split('>')[0]
            if feature_name in self.feature_names:
                feature_contributions[feature_name] = contribution
        
        return {
            'prediction': int(prediction),
            'prediction_proba': prediction_proba.tolist(),
            'feature_contributions': feature_contributions,
            'risk_level': self._get_risk_level(prediction),
            'confidence': float(prediction_proba[prediction]),
            'explanation_text': self._generate_explanation_text(feature_contributions, prediction)
        }
    
    def get_combined_explanation(self, patient_data, model_type='random_forest'):
        """
        Get both SHAP and LIME explanations.
        
        Args:
            patient_data: Dictionary with patient features
            model_type: Model to use for explanation
        
        Returns:
            Dictionary with both SHAP and LIME explanations
        """
        shap_exp = self.get_shap_explanation(patient_data, model_type)
        lime_exp = self.get_lime_explanation(patient_data, model_type)
        
        return {
            'model_type': model_type,
            'shap': shap_exp,
            'lime': lime_exp,
            'patient_data': patient_data
        }
    
    def _prepare_input(self, patient_data):
        """Prepare patient data for model input."""
        # Create DataFrame with correct feature order
        df = pd.DataFrame([patient_data], columns=self.feature_names)
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        return X_scaled
    
    def _get_risk_level(self, prediction):
        """Convert prediction to risk level string."""
        risk_levels = {0: 'Low', 1: 'Medium', 2: 'High'}
        return risk_levels.get(prediction, 'Unknown')
    
    def _generate_explanation_text(self, feature_contributions, prediction):
        """Generate human-readable explanation text."""
        risk_level = self._get_risk_level(prediction)
        
        # Get top contributing features
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        
        explanation = f"The model predicts {risk_level} risk. "
        explanation += "Key factors contributing to this prediction: "
        
        factors = []
        for feature, contribution in sorted_features:
            direction = "increases" if contribution > 0 else "decreases"
            factors.append(f"{feature} {direction} risk")
        
        explanation += ", ".join(factors) + "."
        
        return explanation
    
    def visualize_shap_waterfall(self, patient_data, model_type='random_forest'):
        """
        Create SHAP waterfall plot and return as base64 image.
        
        Args:
            patient_data: Dictionary with patient features
            model_type: Model to use
        
        Returns:
            Base64 encoded image string
        """
        X_input = self._prepare_input(patient_data)
        
        # Select model and create explainer
        if model_type == 'random_forest':
            explainer = shap.TreeExplainer(self.rf_model)
        elif model_type == 'logistic_regression':
            explainer = shap.LinearExplainer(self.lr_model, self.X_background)
        else:
            explainer = shap.KernelExplainer(self.nn_model.predict_proba, self.X_background)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_input)
        
        # Create waterfall plot
        plt.figure(figsize=(10, 6))
        
        if isinstance(shap_values, list):
            # Multi-class: use values for predicted class
            if model_type == 'neural_network':
                prediction = int(self.nn_model.predict(X_input)[0])
            elif model_type == 'random_forest':
                prediction = int(self.rf_model.predict(X_input)[0])
            else:
                prediction = int(self.lr_model.predict(X_input)[0])
            shap_values_plot = shap_values[prediction][0]
        else:
            shap_values_plot = shap_values[0]
        
        # Create bar plot of feature importance
        feature_importance = dict(zip(self.feature_names, shap_values_plot))
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        features = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]
        colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in values]
        
        plt.barh(features, values, color=colors)
        plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
        plt.title('Feature Importance for Prediction', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64

if __name__ == "__main__":
    # Test the explainer
    print("Testing XAI Explainer...")
    
    explainer = XAIExplainer()
    
    # Test patient data
    test_patient = {
        'age': 55,
        'bmi': 32.5,
        'glucose': 160,
        'blood_pressure': 145,
        'cholesterol': 240,
        'smoking': 1,
        'exercise': 1,
        'family_history': 1
    }
    
    print("\n=== SHAP Explanation ===")
    shap_result = explainer.get_shap_explanation(test_patient, 'random_forest')
    print(f"Prediction: {shap_result['risk_level']} Risk")
    print(f"Confidence: {shap_result['confidence']:.2%}")
    print("\nFeature Importance:")
    for feature, importance in list(shap_result['feature_importance'].items())[:5]:
        print(f"  {feature}: {importance:.4f}")
    
    print("\n=== LIME Explanation ===")
    lime_result = explainer.get_lime_explanation(test_patient, 'random_forest')
    print(f"Prediction: {lime_result['risk_level']} Risk")
    print(f"Confidence: {lime_result['confidence']:.2%}")
    print(f"\nExplanation: {lime_result['explanation_text']}")
    
    print("\n✅ XAI Explainer test completed successfully!")
