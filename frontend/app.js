// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const patientForm = document.getElementById('patientForm');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');

// Tab Elements
const tabs = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.tab-content');

// Result Elements
const riskLevel = document.getElementById('riskLevel');
const confidenceScore = document.getElementById('confidenceScore');
const modelBadge = document.getElementById('modelBadge');
const probLow = document.getElementById('probLow');
const probMedium = document.getElementById('probMedium');
const probHigh = document.getElementById('probHigh');
const probLowBar = document.getElementById('probLowBar');
const probMediumBar = document.getElementById('probMediumBar');
const probHighBar = document.getElementById('probHighBar');

// XAI Elements
const shapChart = document.getElementById('shapChart');
const limeExplanation = document.getElementById('limeExplanation');
const featureList = document.getElementById('featureList');
const comparisonGrid = document.getElementById('comparisonGrid');

// Tab Switching
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const targetTab = tab.dataset.tab;

        // Update active tab
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        // Update active content
        tabContents.forEach(content => {
            content.classList.add('hidden');
            content.classList.remove('active');
        });
        const targetContent = document.getElementById(`${targetTab}Tab`);
        targetContent.classList.remove('hidden');
        targetContent.classList.add('active');
    });
});

// Form Submission
patientForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Get form data
    const formData = new FormData(patientForm);
    const patientData = {
        age: parseFloat(formData.get('age')),
        bmi: parseFloat(formData.get('bmi')),
        glucose: parseFloat(formData.get('glucose')),
        blood_pressure: parseFloat(formData.get('blood_pressure')),
        cholesterol: parseFloat(formData.get('cholesterol')),
        smoking: parseInt(formData.get('smoking')),
        exercise: parseInt(formData.get('exercise')),
        family_history: parseInt(formData.get('family_history'))
    };

    const modelType = formData.get('model_type');

    // Show loading
    loading.classList.remove('hidden');
    loading.classList.add('active');
    resultsSection.classList.add('hidden');
    resultsSection.classList.remove('active');

    try {
        // Get prediction and model comparison (skip broken explain endpoint)
        await Promise.all([
            getPrediction(patientData, modelType),
            getModelComparison(patientData)
        ]);

        // Show results
        resultsSection.classList.remove('hidden');
        resultsSection.classList.add('active');

    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing patient data. Please ensure the API server is running.');
    } finally {
        loading.classList.add('hidden');
        loading.classList.remove('active');
    }
});

// Get Prediction
async function getPrediction(patientData, modelType) {
    const response = await fetch(`${API_BASE_URL}/predict?model_type=${modelType}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(patientData)
    });

    if (!response.ok) {
        throw new Error('Prediction failed');
    }

    const data = await response.json();
    displayPrediction(data);
}

// Get Explanation
async function getExplanation(patientData, modelType) {
    const response = await fetch(`${API_BASE_URL}/explain?model_type=${modelType}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(patientData)
    });

    if (!response.ok) {
        throw new Error('Explanation failed');
    }

    const data = await response.json();
    displayExplanation(data);
}

// Get Model Comparison
async function getModelComparison(patientData) {
    const response = await fetch(`${API_BASE_URL}/models/compare`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(patientData)
    });

    if (!response.ok) {
        throw new Error('Comparison failed');
    }

    const data = await response.json();
    displayComparison(data);
}

// Display Prediction
function displayPrediction(data) {
    // Update risk level
    riskLevel.textContent = `${data.risk_level} Risk`;
    riskLevel.className = 'risk-level';
    riskLevel.classList.add(`risk-${data.risk_level.toLowerCase()}`);

    // Update confidence
    confidenceScore.textContent = `${(data.confidence * 100).toFixed(1)}%`;

    // Update model badge
    modelBadge.textContent = formatModelName(data.model_type);

    // Update probabilities
    const probabilities = data.probabilities;
    probLow.textContent = `${(probabilities.Low * 100).toFixed(1)}%`;
    probMedium.textContent = `${(probabilities.Medium * 100).toFixed(1)}%`;
    probHigh.textContent = `${(probabilities.High * 100).toFixed(1)}%`;

    // Animate probability bars
    setTimeout(() => {
        probLowBar.style.width = `${probabilities.Low * 100}%`;
        probMediumBar.style.width = `${probabilities.Medium * 100}%`;
        probHighBar.style.width = `${probabilities.High * 100}%`;
    }, 100);
}

// Display Explanation
function displayExplanation(data) {
    // Display SHAP chart
    if (data.feature_importance_chart) {
        shapChart.innerHTML = `<img src="data:image/png;base64,${data.feature_importance_chart}" alt="SHAP Feature Importance">`;
    }

    // Display LIME explanation
    if (data.lime_explanation && data.lime_explanation.explanation_text) {
        limeExplanation.textContent = data.lime_explanation.explanation_text;
    }

    // Display feature importance list
    if (data.shap_explanation && data.shap_explanation.feature_importance) {
        const features = data.shap_explanation.feature_importance;
        const featureEntries = Object.entries(features).slice(0, 8); // Top 8 features

        featureList.innerHTML = featureEntries.map(([name, value]) => {
            const valueClass = value > 0 ? 'positive' : 'negative';
            const arrow = value > 0 ? '↑' : '↓';
            return `
                <li class="feature-item">
                    <span class="feature-name">${formatFeatureName(name)}</span>
                    <span class="feature-value ${valueClass}">${arrow} ${Math.abs(value).toFixed(4)}</span>
                </li>
            `;
        }).join('');
    }
}

// Display Model Comparison
function displayComparison(data) {
    const predictions = data.predictions;

    comparisonGrid.innerHTML = Object.entries(predictions).map(([modelType, prediction]) => {
        const riskClass = `risk-${prediction.risk_level.toLowerCase()}`;
        return `
            <div class="comparison-card">
                <div class="comparison-model-name">${formatModelName(modelType)}</div>
                <div class="comparison-risk ${riskClass}">${prediction.risk_level}</div>
                <div class="comparison-confidence">${(prediction.confidence * 100).toFixed(1)}% confidence</div>
            </div>
        `;
    }).join('');
}

// Utility Functions
function formatModelName(modelType) {
    const names = {
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'neural_network': 'Neural Network'
    };
    return names[modelType] || modelType;
}

function formatFeatureName(name) {
    const names = {
        'age': 'Age',
        'bmi': 'BMI',
        'glucose': 'Glucose Level',
        'blood_pressure': 'Blood Pressure',
        'cholesterol': 'Cholesterol',
        'smoking': 'Smoking Status',
        'exercise': 'Exercise Frequency',
        'family_history': 'Family History'
    };
    return names[name] || name;
}

// Initialize - Check API health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('✅ API server is running');
        }
    } catch (error) {
        console.warn('⚠️ API server is not running. Please start the backend server.');
    }
}

// Check API on load
checkAPIHealth();
