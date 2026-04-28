"""Generate a synthetic diabetes_prediction_dataset.csv with the expected schema."""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 100000

# Create synthetic data with the expected schema
data = {
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'age': np.random.uniform(18, 80, n_samples),
    'hypertension': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    'heart_disease': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    'smoking_history': np.random.choice(['never', 'former', 'current', 'Not applicable'], n_samples),
    'bmi': np.random.uniform(10, 50, n_samples),
    'HbA1c_level': np.random.uniform(4, 14, n_samples),
    'blood_glucose_level': np.random.uniform(80, 300, n_samples),
    'diabetes': np.random.choice([0, 1], n_samples, p=[0.88, 0.12]),  # ~12% diabetes prevalence
}

df = pd.DataFrame(data)

# Save to CSV
output_path = Path(__file__).parent / "diabetes_prediction_dataset.csv"
df.to_csv(output_path, index=False)
print(f"✓ Generated synthetic dataset: {output_path}")
print(f"  Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
