import joblib
import pandas as pd


# Load trained model
model = joblib.load("aeb_random_forest_model.pkl")


print("===================================")
print("       AEB COLLISION PREDICTION")
print("===================================")


# Get inputs
vehicle_speed = float(input("Vehicle speed (km/h): "))
mass = float(input("Vehicle mass (kg): "))
distance = float(input("Distance to obstacle (m): "))
relative_speed = float(input("Relative speed (km/h): "))
friction = float(input("Friction coefficient: "))

road_environment = input(
    "Road environment (Dry/Wet/Icy): "
)

weather_condition = input(
    "Weather condition (Clear/Rain/Fog): "
)


# Create input DataFrame
new_input = pd.DataFrame([{
    "vehicle_speed": vehicle_speed,
    "mass": mass,
    "distance": distance,
    "relative_speed": relative_speed,
    "friction": friction,
    "road_environment": road_environment,
    "weather_condition": weather_condition
}])


# Predict collision risk
prediction = model.predict(new_input)


# Display result
print("\n===================================")
print("           PREDICTION")
print("===================================")

if prediction[0] == 1:
    print("Collision Risk: YES")
else:
    print("Collision Risk: NO")