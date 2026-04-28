"""Create mock advanced diabetes model artifacts as a workaround while the full pipeline runs."""

import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Create the artifacts directory
artifacts_dir = Path("DIABETES PREDICTION/outputs/advanced_pipeline/artifacts")
artifacts_dir.mkdir(parents=True, exist_ok=True)

# Create a mock RandomForestClassifier
# Train on simple synthetic data
X_mock = np.random.randn(100, 7)
y_mock = np.random.randint(0, 2, 100)

mock_model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
mock_model.fit(X_mock, y_mock)

# Create a mock scaler
mock_scaler = StandardScaler()
mock_scaler.fit(X_mock)

# Save the mock artifacts
joblib.dump(mock_model, artifacts_dir / "tuned_random_forest.joblib")
joblib.dump(mock_scaler, artifacts_dir / "engineered_feature_scaler.joblib")

# Create mock metadata
metadata = {
    "status": "mock_generated",
    "note": "Generated as temporary placeholder while full pipeline runs",
    "feature_names": ["gender", "hypertension", "heart_disease", "smoking_history", "bmi_age", "glucose_hba1c", "hypertension_heart"],
    "best_model": "RandomForestClassifier",
    "best_params": {"n_estimators": 10, "max_depth": 5, "random_state": 42},
}

metadata_path = Path("DIABETES PREDICTION/outputs/advanced_pipeline/run_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print("✓ Mock artifacts created in:")
print(f"  - {artifacts_dir / 'tuned_random_forest.joblib'}")
print(f"  - {artifacts_dir / 'engineered_feature_scaler.joblib'}")
print(f"  - {metadata_path}")
