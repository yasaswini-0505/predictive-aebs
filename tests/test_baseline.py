from pathlib import Path
import joblib
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_datasets_exist():
    """Verify that both primary and independent test datasets exist with valid schemas and no nulls."""
    train_path = PROJECT_ROOT / "aeb_dataset.csv"
    test_path = PROJECT_ROOT / "aeb_new_dataset.csv"

    assert train_path.exists(), f"Missing primary dataset: {train_path}"
    assert test_path.exists(), f"Missing independent dataset: {test_path}"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    expected_cols = [
        "vehicle_speed",
        "mass",
        "distance",
        "relative_speed",
        "friction",
        "road_environment",
        "weather_condition",
        "collision_risk"
    ]

    assert list(df_train.columns) == expected_cols, "Dataset columns do not match expected schema"
    assert list(df_test.columns) == expected_cols, "Test dataset columns do not match expected schema"

    assert len(df_train) == 10000, f"Expected 10,000 rows in aeb_dataset.csv, found {len(df_train)}"
    assert len(df_test) == 1000, f"Expected 1,000 rows in aeb_new_dataset.csv, found {len(df_test)}"

    assert df_train.isnull().sum().sum() == 0, "Null values detected in training dataset"
    assert df_test.isnull().sum().sum() == 0, "Null values detected in test dataset"


def test_saved_model_loads_and_predicts():
    """Verify that the persisted Random Forest pipeline loads and outputs calibrated collision probabilities."""
    model_path = PROJECT_ROOT / "aeb_random_forest_model.pkl"
    assert model_path.exists(), f"Trained model artifact not found: {model_path}"

    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "Loaded model has no predict method"
    assert hasattr(model, "predict_proba"), "Loaded model has no predict_proba method"

    # Test single-row inference
    sample_df = pd.DataFrame([{
        "vehicle_speed": 60.0,
        "mass": 1500.0,
        "distance": 20.0,
        "relative_speed": 40.0,
        "friction": 0.8,
        "road_environment": "Dry",
        "weather_condition": "Clear"
    }])

    pred = model.predict(sample_df)
    prob = model.predict_proba(sample_df)

    assert pred[0] in [0, 1], "Prediction must be binary (0 or 1)"
    assert prob.shape == (1, 2), "Prediction probabilities must have shape (1, 2)"
    assert 0.0 <= prob[0, 1] <= 1.0, "Collision probability must be in [0, 1]"
    assert pytest.approx(prob[0, 0] + prob[0, 1], abs=1e-5) == 1.0, "Class probabilities must sum to 1.0"

