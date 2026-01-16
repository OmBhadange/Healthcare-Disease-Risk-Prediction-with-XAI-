import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_healthcare_dataset(n_samples=2000, random_state=42):
    """
    Generate synthetic healthcare dataset with realistic patient attributes.
    
    Features:
    - age: 20-80 years
    - bmi: 18-45 (Body Mass Index)
    - glucose: 70-200 mg/dL
    - blood_pressure: 90-180 mmHg (systolic)
    - cholesterol: 150-300 mg/dL
    - smoking: 0 (non-smoker) or 1 (smoker)
    - exercise: 0-5 (days per week)
    - family_history: 0 (no) or 1 (yes)
    
    Target:
    - disease_risk: 0 (Low), 1 (Medium), 2 (High)
    """
    np.random.seed(random_state)
    
    # Generate base features
    age = np.random.normal(50, 15, n_samples).clip(20, 80)
    bmi = np.random.normal(27, 5, n_samples).clip(18, 45)
    glucose = np.random.normal(110, 25, n_samples).clip(70, 200)
    blood_pressure = np.random.normal(125, 20, n_samples).clip(90, 180)
    cholesterol = np.random.normal(210, 35, n_samples).clip(150, 300)
    smoking = np.random.binomial(1, 0.25, n_samples)
    exercise = np.random.poisson(2.5, n_samples).clip(0, 5)
    family_history = np.random.binomial(1, 0.35, n_samples)
    
    # Calculate risk score based on medical factors
    risk_score = (
        (age - 20) / 60 * 0.15 +  # Age factor
        (bmi - 18) / 27 * 0.20 +  # BMI factor
        (glucose - 70) / 130 * 0.25 +  # Glucose factor (highest weight)
        (blood_pressure - 90) / 90 * 0.15 +  # BP factor
        (cholesterol - 150) / 150 * 0.10 +  # Cholesterol factor
        smoking * 0.10 +  # Smoking penalty
        (5 - exercise) / 5 * 0.05 +  # Exercise factor (inverse)
        family_history * 0.15  # Family history factor
    )
    
    # Add some randomness to make it more realistic
    risk_score += np.random.normal(0, 0.1, n_samples)
    risk_score = risk_score.clip(0, 1)
    
    # Convert to categorical risk levels
    disease_risk = np.zeros(n_samples, dtype=int)
    disease_risk[risk_score > 0.35] = 1  # Medium risk
    disease_risk[risk_score > 0.65] = 2  # High risk
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age.round(1),
        'bmi': bmi.round(1),
        'glucose': glucose.round(1),
        'blood_pressure': blood_pressure.round(1),
        'cholesterol': cholesterol.round(1),
        'smoking': smoking,
        'exercise': exercise,
        'family_history': family_history,
        'disease_risk': disease_risk
    })
    
    return df

if __name__ == "__main__":
    # Generate dataset
    print("Generating healthcare dataset...")
    df = generate_healthcare_dataset(n_samples=2000)
    
    # Save to CSV
    output_path = "healthcare_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    
    # Display statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {len(df)}")
    print(f"\nRisk distribution:")
    print(df['disease_risk'].value_counts().sort_index())
    print(f"\nRisk percentages:")
    print(df['disease_risk'].value_counts(normalize=True).sort_index() * 100)
    
    print("\n=== Feature Statistics ===")
    print(df.describe())
    
    print("\n=== Sample Records ===")
    print(df.head(10))
