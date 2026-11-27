"""
Predictive Random Forest Training and Feature Importance Analysis.

Trains a predictive Random Forest Classifier utilizing advanced dynamic perception features:
- Lead vehicle deceleration
- Brake-light / braking intent detection
- Relative acceleration
- Projected acceleration-corrected TTC
- Surface friction uncertainty
- Vehicle mass, speeds, and environmental attributes

Outputs:
- Saved model: aeb_predictive_rf_model.pkl (completely separate from existing model)
- Validation metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Feature importance analysis ranking predictive contribution
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Dict, List, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parent

CATEGORICAL_FEATURES = ["road_environment", "weather_condition"]
NUMERICAL_FEATURES = [
    "vehicle_speed",
    "distance",
    "relative_speed",
    "lead_deceleration",
    "brake_light_detected",
    "relative_acceleration",
    "time_to_collision",
    "projected_ttc",
    "friction_estimate",
    "mass"
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET_COLUMN = "collision_risk"


def train_predictive_model(
    dataset_csv: str = "aeb_predictive_dataset.csv",
    out_model_path: str = "aeb_predictive_rf_model.pkl"
) -> Tuple[Pipeline, Dict[str, float]]:
    data_path = PROJECT_ROOT / dataset_csv
    if not data_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {dataset_csv}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train set: {len(X_train)} samples | Validation hold-out: {len(X_val)} samples")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(categories=[["Dry", "Icy", "Wet"], ["Clear", "Fog", "Rain"]], handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERICAL_FEATURES)
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    print("Training Predictive Random Forest Classifier...")
    pipeline.fit(X_train, y_train)

    # Validation evaluation
    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    cm = confusion_matrix(y_val, y_pred)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    }

    print("\n" + "=" * 60)
    print("      PREDICTIVE ML MODEL VALIDATION REPORT")
    print("=" * 60)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print(f"ROC-AUC   : {auc * 100:.2f}%")
    print("-" * 60)
    print("Confusion Matrix:")
    print(cm)
    print("=" * 60)

    # Feature Importance Extraction
    cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = cat_feature_names + NUMERICAL_FEATURES
    importances = pipeline.named_steps["classifier"].feature_importances_

    fi_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print("\nPredictive Feature Importance Ranking:")
    print("-" * 60)
    for idx, row in fi_df.iterrows():
        print(f"  {row['feature']:<28} : {row['importance'] * 100:>6.2f}%")
    print("-" * 60)

    # Save model
    save_path = PROJECT_ROOT / out_model_path
    joblib.dump(pipeline, save_path)
    print(f"Successfully saved predictive model to: {save_path}")

    return pipeline, metrics


if __name__ == "__main__":
    train_predictive_model()
