"""
Predictive AEB Dataset Generator with Dynamic Temporal Features.

Generates a dedicated synthetic dataset of driving states and trajectory snapshots
containing advanced predictive ADAS features:
1. Lead vehicle deceleration (a_lead in m/s^2)
2. Lead vehicle brake-light / braking-intent indicator (0 or 1)
3. Relative acceleration (a_rel = a_ego - a_lead in m/s^2)
4. Range-rate and acceleration-corrected projected TTC
5. Noisy road friction estimation (sensor uncertainty)
6. Instantaneous headway, speed, closing velocity, mass, road, and weather.

Uses seed=101 strictly separated from benchmark test scenarios (seed=42)
to guarantee zero data leakage.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent


def compute_projected_ttc(gap_m: float, v_rel_ms: float, a_rel_ms2: float) -> float:
    """
    Computes acceleration-corrected projected Time-to-Collision.
    Solves gap - v_rel*t - 0.5*a_rel*t^2 = 0 for positive real t.
    """
    if gap_m <= 0:
        return 0.0

    # If acceleration is negligible, fallback to standard TTC
    if abs(a_rel_ms2) < 0.05:
        if v_rel_ms > 0.05:
            return gap_m / v_rel_ms
        return 999.0

    # Quadratic equation: 0.5*a_rel * t^2 + v_rel * t - gap = 0
    # A = 0.5*a_rel, B = v_rel, C = -gap
    A = 0.5 * a_rel_ms2
    B = v_rel_ms
    C = -gap_m
    disc = B**2 - 4 * A * C

    if disc < 0:
        return 999.0

    sqrt_disc = np.sqrt(disc)
    t1 = (-B + sqrt_disc) / (2 * A)
    t2 = (-B - sqrt_disc) / (2 * A)

    # We want the smallest positive solution
    pos_roots = [t for t in (t1, t2) if t > 0.01]
    if pos_roots:
        return float(min(pos_roots))
    return 999.0


def generate_predictive_dataset(
    n_scenarios: int = 500,
    seed: int = 101,
    out_csv: str = "aeb_predictive_dataset.csv"
) -> pd.DataFrame:
    """
    Generates dynamic trajectory snapshots for predictive model training.
    """
    rng = np.random.RandomState(seed)
    records = []

    def get_road_weather(mu: float):
        if mu > 0.65:
            return "Dry", rng.choice(["Clear", "Clear", "Fog"])
        elif mu > 0.28:
            return "Wet", rng.choice(["Rain", "Rain", "Clear"])
        else:
            return "Icy", rng.choice(["Fog", "Rain", "Clear"])

    for sc_idx in range(n_scenarios):
        # Initial parameters
        v_ego_kmh = float(rng.uniform(30.0, 110.0))
        gap_init = float(rng.uniform(10.0, 75.0))
        mu_true = float(rng.choice([0.85, 0.80, 0.70, 0.55, 0.45, 0.25, 0.18]))
        mass = float(rng.uniform(1150.0, 2300.0))
        road, weather = get_road_weather(mu_true)

        # Maneuver archetype
        archetype = rng.choice([
            "safe_cruise", "opening_gap", "gradual_slow",
            "sudden_brake", "emergency_stop", "adverse_slip"
        ], p=[0.20, 0.15, 0.20, 0.20, 0.15, 0.10])

        if archetype == "safe_cruise":
            lead_accel = 0.0
            target_v_kmh = v_ego_kmh
            brake_start_t = 0.0
        elif archetype == "opening_gap":
            lead_accel = float(rng.uniform(1.0, 3.0))
            target_v_kmh = v_ego_kmh + float(rng.uniform(15.0, 35.0))
            brake_start_t = float(rng.uniform(0.1, 0.4))
        elif archetype == "gradual_slow":
            lead_accel = float(rng.uniform(-3.0, -1.0))
            target_v_kmh = float(rng.uniform(15.0, v_ego_kmh * 0.6))
            brake_start_t = float(rng.uniform(0.2, 0.6))
        elif archetype == "sudden_brake":
            lead_accel = float(rng.uniform(-6.5, -4.0))
            target_v_kmh = 0.0
            brake_start_t = float(rng.uniform(0.2, 0.5))
        elif archetype == "emergency_stop":
            lead_accel = float(rng.uniform(-9.0, -7.0))
            target_v_kmh = 0.0
            brake_start_t = float(rng.uniform(0.1, 0.3))
        else:  # adverse_slip
            lead_accel = float(rng.uniform(-5.0, -2.5))
            target_v_kmh = 0.0
            brake_start_t = float(rng.uniform(0.2, 0.5))
            mu_true = float(rng.choice([0.40, 0.22, 0.18]))
            road, weather = get_road_weather(mu_true)

        # Simulate trajectory without ego intervention to extract state hazard progression
        v_ego = v_ego_kmh / 3.6
        v_obs = v_ego  # Cruising at matched speed initially
        v_obs_target = target_v_kmh / 3.6
        gap = gap_init
        dt = 0.02
        sim_time = 0.0
        max_t = 5.0

        sample_interval = 0.06  # Sample snapshots every 60ms
        last_sample_t = -1.0

        while sim_time < max_t:
            # Current lead acceleration
            if sim_time >= brake_start_t:
                if lead_accel < 0:
                    a_lead = lead_accel if v_obs > v_obs_target else 0.0
                elif lead_accel > 0:
                    a_lead = lead_accel if v_obs < v_obs_target else 0.0
                else:
                    a_lead = 0.0
            else:
                a_lead = 0.0

            v_rel = v_ego - v_obs
            v_rel_kmh = v_rel * 3.6
            v_ego_kmh_curr = v_ego * 3.6
            a_ego = 0.0  # Unassisted trajectory
            a_rel = a_ego - a_lead

            # Record snapshot
            if (sim_time - last_sample_t) >= sample_interval and gap > 0.5:
                last_sample_t = sim_time

                # Ground truth hazard evaluation:
                # Stopping distance of lead vehicle:
                if abs(a_lead) > 0.1 and a_lead < 0:
                    s_lead = (v_obs**2) / (2.0 * abs(a_lead))
                elif v_obs <= 0.05:
                    s_lead = 0.0
                else:
                    s_lead = 1000.0  # Cruising / moving away

                # Required stopping distance for ego vehicle (accounting for reaction time 0.2s):
                a_max_ego = min(12000.0 / mass, mu_true * 9.81)
                s_ego_req = (v_ego * 0.20) + (v_ego**2) / (2.0 * a_max_ego)

                # Net gap at stop if ego starts braking at this moment:
                net_gap = gap + s_lead - s_ego_req

                # Calculate projected TTC
                proj_ttc = compute_projected_ttc(gap, v_rel, a_rel)
                classical_ttc = (gap / v_rel) if v_rel > 0.05 else 999.0

                # Ground truth label:
                # 1 = hazardous collision trajectory requiring intervention
                # 0 = safe trajectory
                is_hazard = 0
                if gap <= 1.0:
                    is_hazard = 1
                elif a_lead < -1.5 and net_gap <= 2.5:
                    is_hazard = 1
                elif proj_ttc <= 2.2:
                    is_hazard = 1
                elif classical_ttc <= 2.0 and v_rel > 1.5:
                    is_hazard = 1

                # Sensor noise simulation:
                # 1. Brake-light detection with optical latency and weather reliability
                is_braking_active = bool(a_lead < -0.8 and v_obs > 0.1)
                optical_reliability = 0.99 if weather == "Clear" else (0.92 if weather == "Rain" else 0.85)
                brake_light = 1 if (is_braking_active and rng.random() < optical_reliability) else 0

                # 2. Lead deceleration perception noise (sigma = 0.25 m/s^2)
                lead_accel_perceived = float(np.clip(a_lead + rng.normal(0, 0.25), -10.0, 5.0))

                # 3. Relative acceleration perceived
                rel_accel_perceived = float(0.0 - lead_accel_perceived)

                # 4. Friction estimate uncertainty (sigma = 8%)
                friction_perceived = float(np.clip(mu_true * (1.0 + rng.normal(0, 0.08)), 0.15, 0.95))

                # 5. Distance measurement noise (sigma = 0.15m)
                distance_perceived = float(max(0.1, gap + rng.normal(0, 0.15)))

                # 6. Relative speed measurement noise (sigma = 0.15 km/h)
                rel_speed_perceived = float(max(0.0, v_rel_kmh + rng.normal(0, 0.15)))

                records.append({
                    "vehicle_speed": round(v_ego_kmh_curr, 2),
                    "distance": round(distance_perceived, 2),
                    "relative_speed": round(rel_speed_perceived, 2),
                    "lead_deceleration": round(lead_accel_perceived, 2),
                    "brake_light_detected": int(brake_light),
                    "relative_acceleration": round(rel_accel_perceived, 2),
                    "time_to_collision": round(min(999.0, classical_ttc), 2),
                    "projected_ttc": round(min(999.0, proj_ttc), 2),
                    "friction_estimate": round(friction_perceived, 2),
                    "mass": round(mass, 1),
                    "road_environment": road,
                    "weather_condition": weather,
                    "collision_risk": int(is_hazard)
                })

            # Integrate forward
            delta_v_lead = a_lead * dt
            if a_lead < 0:
                if (v_obs + delta_v_lead) <= v_obs_target:
                    v_obs = v_obs_target
                else:
                    v_obs += delta_v_lead
            elif a_lead > 0:
                if (v_obs + delta_v_lead) >= v_obs_target:
                    v_obs = v_obs_target
                else:
                    v_obs += delta_v_lead

            v_obs = max(0.0, v_obs)
            closing = (v_ego - v_obs) * dt
            gap -= closing
            sim_time += dt

            if gap <= 0.0 or (v_ego <= 0.05 and v_obs <= 0.05):
                break

    df = pd.DataFrame(records)
    out_path = PROJECT_ROOT / out_csv
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} predictive training records across {n_scenarios} scenarios.")
    print(f"Class distribution: 0={sum(df.collision_risk==0)}, 1={sum(df.collision_risk==1)} ({df.collision_risk.mean()*100:.1f}% hazards)")
    print(f"Saved dataset to: {out_path}")
    return df


if __name__ == "__main__":
    generate_predictive_dataset(n_scenarios=500, seed=101, out_csv="aeb_predictive_dataset.csv")

