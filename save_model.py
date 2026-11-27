import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# Load original dataset
df = pd.read_csv("aeb_dataset.csv")

# Separate inputs and output
X = df.drop("collision_risk", axis=1)
y = df["collision_risk"]


# Feature types
categorical_features = [
    "road_environment",
    "weather_condition"
]

numerical_features = [
    "vehicle_speed",
    "mass",
    "distance",
    "relative_speed",
    "friction"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", rf)
])


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Train model
model.fit(X_train, y_train)

print("Random Forest trained successfully!")


# Save model
joblib.dump(model, "aeb_random_forest_model.pkl")

print("Model saved as: aeb_random_forest_model.pkl")