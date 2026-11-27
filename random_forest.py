
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load dataset
df = pd.read_csv("aeb_dataset.csv")

# 2. Separate inputs and output
X = df.drop("collision_risk", axis=1)
y = df["collision_risk"]

# 3. Define categorical features
categorical_features = [
    "road_environment",
    "weather_condition"
]

# 4. Preprocessing
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

# 5. Create Random Forest pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

# 6. Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 7. Train the Random Forest
model.fit(X_train, y_train)

print("\nRandom Forest training completed!")

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 10. Display results
print("\n========== RANDOM FOREST RESULTS ==========")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))

