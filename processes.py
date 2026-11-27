
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# 1. Load the dataset
df = pd.read_csv("aeb_dataset.csv")

print("Dataset shape:", df.shape)

# 2. Separate input features and target
X = df.drop("collision_risk", axis=1)
y = df["collision_risk"]

print("\nInput features:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)

# 3. Define categorical and numerical columns
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

# 4. Encode categorical features
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

X_encoded = preprocessor.fit_transform(X)

print("\nEncoded feature matrix shape:", X_encoded.shape)

# 5. Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Display the results
print("\n========== TRAIN / TEST SPLIT ==========")

print("Training input shape:", X_train.shape)
print("Testing input shape:", X_test.shape)

print("Training output shape:", y_train.shape)
print("Testing output shape:", y_test.shape)

print("\nTraining collision distribution:")
print(y_train.value_counts())

print("\nTesting collision distribution:")
print(y_test.value_counts())

print("\nPreprocessing and 80/20 split completed successfully!")

