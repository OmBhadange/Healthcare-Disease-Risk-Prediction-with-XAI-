import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class HealthcarePredictionModels:
    def __init__(self, data_path='healthcare_dataset.csv'):
        """Initialize the model training pipeline."""
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the healthcare dataset."""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        # Separate features and target
        X = df.drop('disease_risk', axis=1)
        y = df['disease_risk']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
        print(f"Features: {list(X.columns)}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_logistic_regression(self):
        """Train Logistic Regression baseline model."""
        print("\n=== Training Logistic Regression ===")
        
        lr_model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        
        lr_model.fit(self.X_train_scaled, self.y_train)
        self.models['logistic_regression'] = lr_model
        
        # Evaluate
        y_pred = lr_model.predict(self.X_test_scaled)
        self._print_evaluation_metrics("Logistic Regression", y_pred)
        
        return lr_model
    
    def train_random_forest(self):
        """Train Random Forest Classifier with hyperparameter tuning."""
        print("\n=== Training Random Forest ===")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf_base = RandomForestClassifier(random_state=42)
        
        print("Performing GridSearchCV for hyperparameter tuning...")
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        
        rf_model = grid_search.best_estimator_
        self.models['random_forest'] = rf_model
        
        # Evaluate
        y_pred = rf_model.predict(self.X_test_scaled)
        self._print_evaluation_metrics("Random Forest", y_pred)
        
        return rf_model
    
    def train_neural_network(self):
        """Train Neural Network (Multi-Layer Perceptron) using scikit-learn."""
        print("\n=== Training Neural Network (MLP) ===")
        
        # Build model using scikit-learn's MLPClassifier
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10,
            verbose=True
        )
        
        # Train
        model.fit(self.X_train_scaled, self.y_train)
        
        self.models['neural_network'] = model
        
        # Evaluate
        y_pred = model.predict(self.X_test_scaled)
        self._print_evaluation_metrics("Neural Network", y_pred)
        
        # Plot training history
        self._plot_training_history(model)
        
        return model
    
    def _print_evaluation_metrics(self, model_name, y_pred):
        """Print evaluation metrics for a model."""
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"\n{model_name} Performance:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Low Risk', 'Medium Risk', 'High Risk']))
        
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
    
    def _plot_training_history(self, model):
        """Plot training history for neural network."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Loss plot
        ax.plot(model.loss_curve_, label='Training Loss', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Neural Network Training Loss', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('nn_training_history.png', dpi=150, bbox_inches='tight')
        print("Training history plot saved to nn_training_history.png")
        plt.close()
    
    def save_models(self):
        """Save all trained models."""
        print("\n=== Saving Models ===")
        
        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')
        print("Saved: scaler.pkl")
        
        # Save Logistic Regression
        joblib.dump(self.models['logistic_regression'], 'models/logistic_regression.pkl')
        print("Saved: logistic_regression.pkl")
        
        # Save Random Forest
        joblib.dump(self.models['random_forest'], 'models/random_forest.pkl')
        print("Saved: random_forest.pkl")
        
        # Save Neural Network
        joblib.dump(self.models['neural_network'], 'models/neural_network.pkl')
        print("Saved: neural_network.pkl")
        
        print("\nAll models saved successfully!")
    
    def compare_models(self):
        """Compare all three models."""
        print("\n=== Model Comparison ===")
        
        results = []
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test_scaled)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            results.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{accuracy:.4f}",
                'Precision': f"{precision:.4f}",
                'Recall': f"{recall:.4f}",
                'F1-Score': f"{f1:.4f}"
            })
        
        comparison_df = pd.DataFrame(results)
        print(comparison_df.to_string(index=False))
        
        return comparison_df

if __name__ == "__main__":
    # Create models directory
    import os
    os.makedirs('models', exist_ok=True)
    
    # Initialize trainer
    trainer = HealthcarePredictionModels('healthcare_dataset.csv')
    
    # Load and preprocess data
    trainer.load_and_preprocess_data()
    
    # Train all models
    trainer.train_logistic_regression()
    trainer.train_random_forest()
    trainer.train_neural_network()
    
    # Compare models
    trainer.compare_models()
    
    # Save models
    trainer.save_models()
    
    print("\n✅ Model training completed successfully!")
