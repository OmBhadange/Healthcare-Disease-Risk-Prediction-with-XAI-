# 🏥 Healthcare Disease Risk Prediction with Explainable AI

An intelligent healthcare system that predicts disease risk using **Machine Learning** and **Deep Learning** algorithms, integrated with **Explainable AI (XAI)** techniques to provide transparent and interpretable predictions for clinical decision support.

## 🌟 Features

- **Multiple ML/DL Models**: Logistic Regression, Random Forest, and Neural Networks
- **Explainable AI**: SHAP and LIME explanations for transparent predictions
- **Modern Web Interface**: Beautiful, responsive UI with real-time visualizations
- **REST API**: FastAPI-based backend with automatic documentation
- **Model Comparison**: Compare predictions across all three models
- **Feature Importance**: Visual representation of factors influencing predictions

## 🛠 Tech Stack

### Backend
- **Python 3.8+**
- **FastAPI** - Modern web framework for building APIs
- **Scikit-learn** - Machine Learning models
- **TensorFlow/Keras** - Deep Learning (Neural Networks)
- **SHAP** - SHapley Additive Explanations
- **LIME** - Local Interpretable Model-agnostic Explanations

### Frontend
- **HTML5/CSS3** - Modern, responsive design
- **JavaScript (ES6+)** - Dynamic interactions
- **Glassmorphism UI** - Premium design aesthetics

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 🚀 Installation & Setup

### 1. Clone or Navigate to Project Directory

```bash
cd "/Users/omavinashbhadange/Clg Stuff/ML AND DL project /untitled folder"
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate Dataset

```bash
cd backend
python data_generator.py
```

This will create a synthetic healthcare dataset (`healthcare_dataset.csv`) with 2000 patient records.

### 5. Train Models

```bash
python model_training.py
```

This will:
- Train Logistic Regression, Random Forest, and Neural Network models
- Evaluate model performance
- Save trained models to the `models/` directory
- Generate training visualizations

**Expected Output:**
- Accuracy > 80% for all models
- Model files saved in `backend/models/`

### 6. Start the API Server

```bash
cd backend
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

**API Documentation:** Visit `http://localhost:8000/docs` for interactive API documentation

### 7. Open the Web Interface

Open `frontend/index.html` in your web browser:

```bash
# On macOS
open frontend/index.html

# On Linux
xdg-open frontend/index.html

# On Windows
start frontend/index.html
```

## 📊 Usage

### Web Interface

1. **Enter Patient Data**: Fill in the patient health information form
   - Age, BMI, Glucose level, Blood pressure, Cholesterol
   - Smoking status, Exercise frequency, Family history

2. **Select AI Model**: Choose from:
   - Random Forest (Recommended)
   - Neural Network
   - Logistic Regression

3. **Analyze Risk**: Click "Analyze Risk" to get predictions

4. **View Results**: Explore three tabs:
   - **Prediction**: Risk level, confidence score, and probabilities
   - **XAI Explanation**: SHAP feature importance and LIME explanations
   - **Model Comparison**: Compare predictions across all models

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Predict Risk
```bash
curl -X POST http://localhost:8000/predict?model_type=random_forest \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "bmi": 28.5,
    "glucose": 140,
    "blood_pressure": 130,
    "cholesterol": 220,
    "smoking": 0,
    "exercise": 2,
    "family_history": 1
  }'
```

#### Get Explanation
```bash
curl -X POST http://localhost:8000/explain?model_type=random_forest \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "bmi": 28.5,
    "glucose": 140,
    "blood_pressure": 130,
    "cholesterol": 220,
    "smoking": 0,
    "exercise": 2,
    "family_history": 1
  }'
```

#### Compare Models
```bash
curl -X POST http://localhost:8000/models/compare \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "bmi": 28.5,
    "glucose": 140,
    "blood_pressure": 130,
    "cholesterol": 220,
    "smoking": 0,
    "exercise": 2,
    "family_history": 1
  }'
```

## 🧠 How It Works

### 1. Data Processing
Patient health data is normalized using StandardScaler to ensure all features are on the same scale.

### 2. Model Prediction
Three models provide predictions:
- **Logistic Regression**: Fast baseline model
- **Random Forest**: Ensemble method with high accuracy
- **Neural Network**: Deep learning for complex patterns

### 3. Explainable AI
- **SHAP**: Calculates feature importance using game theory (Shapley values)
- **LIME**: Creates local explanations by perturbing input features

### 4. Risk Assessment
Predictions are categorized into three risk levels:
- **Low Risk** (0): Minimal disease risk
- **Medium Risk** (1): Moderate disease risk
- **High Risk** (2): High disease risk

## 📁 Project Structure

```
.
├── backend/
│   ├── data_generator.py      # Generate synthetic healthcare dataset
│   ├── model_training.py      # Train ML/DL models
│   ├── xai_explainer.py       # SHAP and LIME explanations
│   ├── models.py              # Pydantic data models
│   ├── main.py                # FastAPI application
│   ├── healthcare_dataset.csv # Generated dataset
│   └── models/                # Trained model files
│       ├── scaler.pkl
│       ├── logistic_regression.pkl
│       ├── random_forest.pkl
│       └── neural_network.h5
├── frontend/
│   ├── index.html             # Web interface
│   ├── styles.css             # Modern styling
│   └── app.js                 # JavaScript logic
└── requirements.txt           # Python dependencies
```

## 🎯 Model Performance

All models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall

Expected performance: **>80% accuracy** across all models

## 🌍 Real-World Applications

- **Clinical Decision Support**: Assist doctors in risk assessment
- **Preventive Healthcare**: Early disease detection and screening
- **Hospital Risk Assessment**: Prioritize high-risk patients
- **Health Insurance**: Risk-based premium calculation
- **AI-Assisted Diagnostics**: Transparent AI reasoning for medical decisions

## 🔬 XAI Techniques Explained

### SHAP (SHapley Additive Explanations)
- Based on game theory
- Provides global and local feature importance
- Shows how each feature contributes to the prediction
- Consistent and mathematically rigorous

### LIME (Local Interpretable Model-agnostic Explanations)
- Model-agnostic approach
- Creates local linear approximations
- Explains individual predictions in human-readable format
- Works with any black-box model

## 🚀 Innovation & Impact

✅ **Combines predictive accuracy with transparency**  
✅ **Builds trust in AI-based medical systems**  
✅ **Helps doctors understand model decisions**  
✅ **Reduces risk of biased or unexplained predictions**  
✅ **Highly relevant to real-world healthcare adoption**

## 📝 License

This project is created for educational and research purposes.

## 👨‍💻 Author

Healthcare AI Research Project

## 🎯 One-Line Pitch

*"An AI-driven healthcare system that predicts disease risk and explains its decisions using Explainable AI techniques."*

---

**Note**: This system uses synthetic data for demonstration purposes. For real-world deployment, use validated medical datasets and consult with healthcare professionals.
