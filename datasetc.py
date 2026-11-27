import random
import csv

# Number of scenarios
NUM_SCENARIOS = 1000

# Simulation constants
g = 9.81
BRAKE_FORCE_MAX = 12000      # Maximum braking force in Newtons
AEB_RESPONSE_TIME = 0.2      # seconds
TIME_STEP = 0.01             # seconds

# Possible categorical values
road_environments = ["Dry", "Wet", "Icy"]
weather_conditions = ["Clear", "Rain", "Fog"]

# Friction ranges corresponding to road conditions
friction_ranges = {
    "Dry": (0.7, 0.9),
    "Wet": (0.4, 0.7),
    "Icy": (0.1, 0.3)
}


def simulate_collision(speed_kmh, mass, distance,
                        relative_speed_kmh, friction,
                        road_environment, weather_condition):

    # Convert speeds from km/h to m/s
    vehicle_speed = speed_kmh / 3.6
    relative_speed = relative_speed_kmh / 3.6

    # Assume obstacle is moving in the same direction
    obstacle_speed = max(vehicle_speed - relative_speed, 0)

    # Calculate braking deceleration
    force_limited_acceleration = BRAKE_FORCE_MAX / mass
    friction_limited_acceleration = friction * g

    acceleration = min(
        force_limited_acceleration,
        friction_limited_acceleration
    )

    # Current gap between vehicle and obstacle
    gap = distance

    # Current vehicle speed
    current_speed = vehicle_speed

    # Simulate for maximum 10 seconds
    time = 0

    while time < 10:

        # AEB response delay
        if time >= AEB_RESPONSE_TIME:
            current_speed = max(
                current_speed - acceleration * TIME_STEP,
                0
            )

        # Distance closed during this time step
        closing_speed = current_speed - obstacle_speed

        gap -= closing_speed * TIME_STEP

        # Collision has occurred
        if gap <= 0:
            return 1

        # Vehicle has stopped safely
        if current_speed <= 0:
            return 0

        time += TIME_STEP

    return 0


def generate_dataset(output_filename="aeb_new_dataset.csv", num_scenarios=NUM_SCENARIOS):
    # Create CSV file
    with open(output_filename, "w", newline="") as file:
        writer = csv.writer(file)
        # Header
        writer.writerow([
            "vehicle_speed",
            "mass",
            "distance",
            "relative_speed",
            "friction",
            "road_environment",
            "weather_condition",
            "collision_risk"
        ])

        for _ in range(num_scenarios):
            # Select road environment
            road_environment = random.choice(road_environments)

            # Select friction based on road environment
            min_friction, max_friction = friction_ranges[road_environment]
            friction = round(
                random.uniform(min_friction, max_friction),
                2
            )

            # Select weather
            weather_condition = random.choice(weather_conditions)

            # Generate vehicle speed
            vehicle_speed = random.randint(20, 100)

            # Generate vehicle mass
            mass = random.randint(1000, 2000)

            # Generate distance to obstacle
            distance = round(
                random.uniform(5, 50),
                2
            )

            # Generate relative speed
            relative_speed = random.uniform(
                5,
                vehicle_speed
            )
            relative_speed = round(relative_speed, 2)

            # Calculate collision risk
            collision_risk = simulate_collision(
                vehicle_speed,
                mass,
                distance,
                relative_speed,
                friction,
                road_environment,
                weather_condition
            )

            # Write row
            writer.writerow([
                vehicle_speed,
                mass,
                distance,
                relative_speed,
                friction,
                road_environment,
                weather_condition,
                collision_risk
            ])

    print("AEB dataset generated successfully!")
    print(f"File name: {output_filename}")
    print("Number of scenarios:", num_scenarios)


if __name__ == "__main__":
    generate_dataset()