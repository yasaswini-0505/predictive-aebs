import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==============================
# 1. Load ORIGINAL dataset
# ==============================

df = pd.read_csv("aeb_dataset.csv")

X = df.drop("collision_risk", axis=1)
y = df["collision_risk"]


# ==============================
# 2. Define features
# ==============================

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


# ==============================
# 3. Preprocessing
# ==============================

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


# ==============================
# 4. Random Forest
# ==============================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", rf)
])


# ==============================
# 5. Original train-test split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==============================
# 6. Train on ORIGINAL training data
# ==============================

model.fit(X_train, y_train)

print("Random Forest trained on original 8,000 rows!")


# ==============================
# 7. Load NEW 1,000-row dataset
# ==============================

new_df = pd.read_csv("aeb_new_dataset.csv")

X_new = new_df.drop("collision_risk", axis=1)
y_new = new_df["collision_risk"]


print("New dataset rows:", len(new_df))


# ==============================
# 8. Predict on NEW dataset
# ==============================

y_new_pred = model.predict(X_new)


# ==============================
# 9. Evaluate predictions
# ==============================

accuracy = accuracy_score(y_new, y_new_pred)

precision = precision_score(
    y_new,
    y_new_pred,
    zero_division=0
)

recall = recall_score(
    y_new,
    y_new_pred,
    zero_division=0
)

f1 = f1_score(
    y_new,
    y_new_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_new,
    y_new_pred
)


# ==============================
# 10. Display results
# ==============================

print("\n========== NEW DATASET RESULTS ==========")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))

print("\n========== CONFUSION MATRIX ==========")
print(cm)