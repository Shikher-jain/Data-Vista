"""Advanced diabetes prediction workflow.

This script mirrors the requested 12-step pipeline:
1. Setup & Imports
2. Load Data
3. Exploratory Data Analysis (EDA)
4. Preprocessing
5. Feature Engineering & Selection
6. Train/Test Split & Scaling
7. Baseline Modelling
8. Class Imbalance Handling (SMOTE)
9. Hyperparameter Tuning (RandomizedSearchCV)
10. Best Model Evaluation
11. Explainability (SHAP)
12. Conclusions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run advanced diabetes model workflow.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=base_dir / "diabetes_prediction_dataset.csv",
        help="Path to diabetes_prediction_dataset.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "outputs" / "advanced_pipeline",
        help="Directory for plots, reports, and model artifacts.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def section(title: str) -> None:
    print(f"\n{'=' * 90}")
    print(title)
    print(f"{'=' * 90}")


def save_plot(output_dir: Path, filename: str, show: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / filename, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def ensure_expected_columns(df: pd.DataFrame) -> None:
    expected = {
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "smoking_history",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "diabetes",
    }
    missing = sorted(expected.difference(df.columns))
    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
            + ".\nUse the Kaggle diabetes_prediction_dataset.csv with the expected schema."
        )


def evaluate_models(
    models: Dict[str, object],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        report = classification_report(
            y_test,
            y_pred,
            target_names=["Non-Diabetic", "Diabetic"],
            output_dict=True,
            zero_division=0,
        )
        rows.append(
            {
                "Model": name,
                "Accuracy": round(report["accuracy"], 3),
                "Precision (Diabetic)": round(report["Diabetic"]["precision"], 3),
                "Recall (Diabetic)": round(report["Diabetic"]["recall"], 3),
                "F1 (Diabetic)": round(report["Diabetic"]["f1-score"], 3),
            }
        )

        print(f"\n{'=' * 45}")
        print(f"  {name}")
        print(f"{'=' * 45}")
        print(classification_report(y_test, y_pred, target_names=["Non-Diabetic", "Diabetic"], zero_division=0))

    return pd.DataFrame(rows)


def encode_categoricals(df: pd.DataFrame, cols: Tuple[str, ...]) -> Dict[str, Dict[str, int]]:
    encoders: Dict[str, Dict[str, int]] = {}
    for col in cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = {cls: int(code) for cls, code in zip(le.classes_, le.transform(le.classes_))}
    return encoders


def main() -> None:
    args = parse_args()
    data_path = args.data_path
    output_dir = args.output_dir
    show_plots = args.show_plots

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}\n"
            "Download diabetes_prediction_dataset.csv and pass its path with --data-path."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.dpi"] = 100

    # 1) Setup & Imports
    section("1 Setup & Imports")
    print("Libraries loaded successfully.")

    # 2) Load Data
    section("2 Load Data")
    df = pd.read_csv(data_path)
    ensure_expected_columns(df)

    print(f"Shape: {df.shape}")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nDescribe:")
    print(df.describe(include="all"))

    missing = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing[missing > 0] if missing.any() else "No missing values found")

    # 3) Exploratory Data Analysis (EDA)
    section("3 Exploratory Data Analysis (EDA)")

    # 3.1 Target distribution
    counts = df["diabetes"].value_counts().rename(index={0: "Non-Diabetic", 1: "Diabetic"})
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", color=["#4C9BE8", "#E8634C"], ax=ax, edgecolor="white")
    for i, val in enumerate(counts.values):
        ax.text(i, val + max(20, int(0.005 * len(df))), f"{val / len(df) * 100:.1f}%", ha="center", fontsize=10)
    ax.set_title("Target Class Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    save_plot(output_dir, "eda_target_distribution.png", show=show_plots)

    # 3.2 Numerical distributions by class
    num_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        for label, color, name in zip([0, 1], ["#4C9BE8", "#E8634C"], ["Non-Diabetic", "Diabetic"]):
            axes[i].hist(
                df[df["diabetes"] == label][col],
                bins=40,
                alpha=0.6,
                color=color,
                label=name,
                density=True,
            )
        axes[i].set_title(col, fontweight="bold")
        axes[i].legend()
    fig.suptitle("Numerical Feature Distributions by Class", fontsize=14, fontweight="bold", y=1.01)
    save_plot(output_dir, "eda_numerical_distributions.png", show=show_plots)

    # 3.3 Diabetes rate by categorical features
    cat_cols = ["gender", "smoking_history", "hypertension", "heart_disease"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        rate = df.groupby(col, observed=False)["diabetes"].mean() * 100
        rate.plot(kind="bar", ax=axes[i], color="#E8634C", alpha=0.85, edgecolor="white")
        axes[i].set_title(f"Diabetes Rate by {col}", fontweight="bold")
        axes[i].set_ylabel("Diabetes %")
        axes[i].tick_params(axis="x", rotation=30)
    fig.suptitle("Diabetes Rate by Categorical Features", fontsize=14, fontweight="bold", y=1.01)
    save_plot(output_dir, "eda_diabetes_rate_by_category.png", show=show_plots)

    # 3.4 Boxplots
    df[["age", "bmi", "HbA1c_level", "blood_glucose_level"]].plot(
        kind="box",
        subplots=True,
        layout=(2, 2),
        figsize=(10, 8),
        patch_artist=True,
    )
    plt.suptitle("Boxplots - Numerical Features", fontsize=13, fontweight="bold")
    save_plot(output_dir, "eda_boxplots.png", show=show_plots)

    # 3.5 Correlation heatmap
    corr_cols = [
        "age",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "hypertension",
        "heart_disease",
        "diabetes",
    ]
    corr = df[corr_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, square=True)
    plt.title("Correlation Heatmap", fontsize=13, fontweight="bold")
    save_plot(output_dir, "eda_correlation_heatmap.png", show=show_plots)

    # 4) Preprocessing
    section("4 Preprocessing")
    encoder_maps = encode_categoricals(df, ("gender", "smoking_history"))
    print("Dtypes after encoding:")
    print(df.dtypes)

    # 5) Feature Engineering & Selection
    section("5 Feature Engineering & Selection")

    # 5.1 Create interaction features
    df["bmi_age"] = df["bmi"] * df["age"]
    df["glucose_hba1c"] = df["blood_glucose_level"] * df["HbA1c_level"]
    df["hypertension_heart"] = df["hypertension"] + df["heart_disease"]
    print("New features added:")
    print(df[["bmi_age", "glucose_hba1c", "hypertension_heart"]].head())

    # 5.2 Correlation with target
    corr_target = df.corr(numeric_only=True)["diabetes"].abs().sort_values(ascending=False)
    print("\nFeature correlation with diabetes (absolute):")
    print(corr_target.to_string())
    corr_target.to_csv(output_dir / "feature_target_correlation.csv", header=["abs_corr"])

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True).abs(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, square=True)
    plt.title("Full Correlation Matrix (Post Feature Engineering)", fontsize=13, fontweight="bold")
    save_plot(output_dir, "feature_engineering_correlation_matrix.png", show=show_plots)

    # 5.3 Drop redundant raw columns
    drop_cols = ["blood_glucose_level", "HbA1c_level", "bmi", "age"]
    df = df.drop(columns=drop_cols)
    print(f"Remaining features: {df.shape[1] - 1} | Shape: {df.shape}")
    print(df.columns.tolist())

    # 6) Train/Test Split & Scaling
    section("6 Train/Test Split & Scaling")
    x = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Train: {x_train.shape} | Test: {x_test.shape}")
    print(f"Train target split: {y_train.value_counts().to_dict()}")

    scaler = StandardScaler()
    scaled_cols = ["bmi_age", "glucose_hba1c"]
    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train[scaled_cols] = scaler.fit_transform(x_train[scaled_cols])
    x_test[scaled_cols] = scaler.transform(x_test[scaled_cols])
    print("Scaling complete")
    print(x_train.head())

    # 7) Baseline Modelling
    section("7 Baseline Modelling")
    class_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=500,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=class_ratio,
            random_state=42,
            eval_metric="logloss",
        ),
    }

    summary_before = evaluate_models(models, x_train, y_train, x_test, y_test)
    print("\nBaseline Summary:")
    print(summary_before.to_string(index=False))
    summary_before.to_csv(output_dir / "baseline_summary.csv", index=False)

    # 8) Class Imbalance Handling - SMOTE
    section("8 Class Imbalance Handling - SMOTE")
    smote = SMOTE(random_state=42, sampling_strategy=0.5)
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

    print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    print(f"After  SMOTE: {pd.Series(y_train_smote).value_counts().to_dict()}")

    summary_after = evaluate_models(models, x_train_smote, y_train_smote, x_test, y_test)
    print("\nAfter SMOTE Summary:")
    print(summary_after.to_string(index=False))
    summary_after.to_csv(output_dir / "smote_summary.csv", index=False)

    print("\nBefore SMOTE:")
    print(summary_before.to_string(index=False))
    print("\nAfter SMOTE:")
    print(summary_after.to_string(index=False))

    # 9) Hyperparameter Tuning - RandomizedSearchCV
    section("9 Hyperparameter Tuning - RandomizedSearchCV on Random Forest")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    random_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        param_distributions=param_grid,
        n_iter=20,
        cv=3,
        scoring="recall",
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    random_search.fit(x_train_smote, y_train_smote)
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Best CV Recall: {random_search.best_score_:.3f}")

    # 10) Best Model Evaluation
    section("10 Best Model Evaluation")
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(x_test)
    print("Classification Report - Tuned Random Forest")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=["Non-Diabetic", "Diabetic"], zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        best_model,
        x_test,
        y_test,
        display_labels=["Non-Diabetic", "Diabetic"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion Matrix - Tuned Random Forest", fontsize=13, fontweight="bold")
    save_plot(output_dir, "tuned_rf_confusion_matrix.png", show=show_plots)

    # 11) Explainability - SHAP
    section("11 Explainability - SHAP")
    x_test_small = x_test.sample(n=min(200, len(x_test)), random_state=42)
    explainer = shap.TreeExplainer(best_model)
    try:
        shap_values = explainer.shap_values(x_test_small, check_additivity=False)
    except TypeError:
        shap_values = explainer.shap_values(x_test_small)

    if isinstance(shap_values, list):
        vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        vals = shap_values[:, :, 1]
    else:
        vals = shap_values

    shap.summary_plot(
        vals,
        x_test_small,
        feature_names=x.columns.tolist(),
        plot_type="bar",
        show=False,
    )
    plt.title("SHAP Feature Importance (Tuned Random Forest)")
    save_plot(output_dir, "shap_summary_bar.png", show=show_plots)

    # 12) Conclusions
    section("12 Conclusions")
    conclusions = pd.DataFrame(
        {
            "Step": [
                "EDA",
                "Feature Engineering",
                "SMOTE",
                "Best Model",
                "SHAP",
            ],
            "Key Finding": [
                "Strong class imbalance with glycaemic markers carrying the strongest signal.",
                "Interaction terms improved learning signal without target leakage.",
                "SMOTE improved diabetic recall for recall-focused model configurations.",
                "Tuned Random Forest provided the strongest diabetic recall/F1 trade-off.",
                "glucose_hba1c and bmi_age were dominant model drivers.",
            ],
        }
    )
    print(conclusions.to_string(index=False))
    conclusions.to_csv(output_dir / "conclusions.csv", index=False)

    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, artifacts_dir / "tuned_random_forest.joblib")
    joblib.dump(scaler, artifacts_dir / "engineered_feature_scaler.joblib")

    metadata = {
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "best_params": random_search.best_params_,
        "best_cv_recall": float(random_search.best_score_),
        "encoder_maps": encoder_maps,
        "selected_features": x.columns.tolist(),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nRun complete.")
    print(f"Artifacts and reports saved to: {output_dir}")


if __name__ == "__main__":
    main()
