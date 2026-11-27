"""Deterministic MetaDrive test scenarios (no randomness at run time)."""
from itertools import product

from config import Scenario

# Twelve named scenarios (Part 8 of the brief). Ego speeds stay <= 80 km/h because MetaDrive
# cuts engine force above 80 km/h for its vehicles.
NAMED = [
    Scenario("S01_stationary", "Stationary lead car", 50, 0, 40, 0.0, 0.0, 0, category="emergency"),
    Scenario("S02_sudden", "Lead brakes suddenly (-6 m/s^2) from 36 km/h", 54, 36, 35, -6.0, 2.0, 0),
    Scenario("S03_gradual", "Lead slows gradually (-2.5 m/s^2) to 25 km/h", 60, 60, 25, -2.5, 2.0, 25),
    Scenario("S04_high_speed", "High-speed approach, lead emergency stop", 80, 80, 35, -8.0, 2.0, 0),
    Scenario("S05_short_gap", "Short following distance (10 m)", 60, 60, 10, -6.0, 2.0, 0),
    Scenario("S06_long_gap", "Long following distance (60 m)", 60, 60, 60, -6.0, 2.0, 0),
    Scenario("S07_heavy", "Heavy ego vehicle (2200 kg)", 60, 60, 25, -6.0, 2.0, 0, mass_kg=2200),
    Scenario("S08_light", "Light ego vehicle (1150 kg)", 60, 60, 25, -6.0, 2.0, 0, mass_kg=1150),
    Scenario("S09_wet", "Wet road, lead brakes hard", 50, 50, 25, -4.5, 2.0, 0, surface="Wet"),
    Scenario("S10_icy", "Icy road, lead brakes", 40, 40, 30, -3.0, 2.0, 0, surface="Icy"),
    Scenario("S11_safe", "Safe following, no lead braking", 60, 60, 50, 0.0, 2.0, 60, category="safe", expect_no_braking=True),
    Scenario("S12_lead_accel", "Slower lead car accelerates away", 60, 50, 30, 1.0, 1.0, 70, category="safe", expect_no_braking=True),
]


def grid():
    """A fixed sweep: archetype x speed x gap x surface (deterministic order)."""
    out = []
    archetypes = [("sudden", -6.0, 0.0), ("emergency", -8.0, 0.0), ("gradual", -2.5, 20.0)]
    for (name, a, target), v, gap, surf in product(archetypes, [40, 60, 80], [15, 30, 45], ["Dry", "Wet"]):
        if surf == "Wet" and v == 80:
            continue
        out.append(Scenario(f"G_{name}_{v}kmh_{gap}m_{surf}", f"{name} braking, {v} km/h, {gap} m, {surf}",
                            v, v, gap, a, 2.0, target, surface=surf))
    for v, gap in product([40, 60, 80], [30, 50]):
        out.append(Scenario(f"G_safe_{v}kmh_{gap}m", f"steady following {v} km/h, {gap} m", v, v, gap, 0.0, 2.0, v,
                            category="safe", expect_no_braking=True))
    return out


def get(name):
    for s in NAMED + grid():
        if s.scenario_id == name:
            return s
    raise KeyError(name)


SUITES = {"named": lambda: list(NAMED), "grid": grid, "all": lambda: list(NAMED) + grid()}
