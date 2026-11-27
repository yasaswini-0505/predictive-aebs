import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# Load dataset
df = pd.read_csv("aeb_dataset.csv")

# Separate inputs and output
X = df.drop("collision_risk", axis=1)
y = df["collision_risk"]


# Features
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

print("Random Forest training completed!")


# Get trained Random Forest
rf_model = model.named_steps["classifier"]

# Get encoded feature names
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

# Get feature importance
importance_values = rf_model.feature_importances_


# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})


# Remove preprocessing prefixes
importance_df["Feature"] = importance_df["Feature"].str.replace(
    "categorical__", "", regex=False
)

importance_df["Feature"] = importance_df["Feature"].str.replace(
    "remainder__", "", regex=False
)


# Create final 7-feature importance dictionary
final_importance = {
    "vehicle_speed": 0,
    "mass": 0,
    "distance": 0,
    "relative_speed": 0,
    "friction": 0,
    "road_environment": 0,
    "weather_condition": 0
}


# Combine one-hot encoded features
for _, row in importance_df.iterrows():

    feature = row["Feature"]
    importance = row["Importance"]

    if feature.startswith("road_environment_"):
        final_importance["road_environment"] += importance

    elif feature.startswith("weather_condition_"):
        final_importance["weather_condition"] += importance

    else:
        final_importance[feature] += importance


# Convert to DataFrame
final_df = pd.DataFrame(
    list(final_importance.items()),
    columns=["Feature", "Importance"]
)


# Sort highest to lowest
final_df = final_df.sort_values(
    by="Importance",
    ascending=False
)


# Add percentage
final_df["Percentage"] = (
    final_df["Importance"] * 100
).round(2)


# Display final results
print("\n========== FINAL FEATURE IMPORTANCE ==========")

print(
    final_df[
        ["Feature", "Importance", "Percentage"]
    ].to_string(index=False)
)


# Bar chart
plt.figure(figsize=(10, 6))

plt.bar(
    final_df["Feature"],
    final_df["Percentage"]
)

plt.xlabel("Input Feature")
plt.ylabel("Importance (%)")
plt.title("Random Forest Feature Importance")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
