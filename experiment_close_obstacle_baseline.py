"""
Experiment Script: Close-Obstacle Investigation and Kinematic Baseline Comparison.

Investigates AEB performance in close-obstacle and low relative speed scenarios,
comparing:
1. Unassisted vehicle (No AEB)
2. Existing ML-based AEBAgent (Random Forest)
3. Physically justified Kinematic Safety Baseline (KinematicAEBAgent)

Does not modify any production code, existing models, or existing tests.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

# Ensure project root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from simulation_engine import SimulationEngine, SimulationOutcome, SimulationResult
from braking_controller import BrakingController, BrakeStage
from aeb_agent import AEBAgent, AEBDecision


class KinematicAEBAgent:
    """
    Experimental physics-based kinematic safety baseline for AEB.
    
    Calculates stopping distance required based on physical vehicle mass,
    actuator force capacity, tire-road friction saturation, and actuator lag:
      a_max = min(F_brake_max / mass, mu * g)
      d_req = v_rel * latency + (v_rel^2) / (2 * a_max)
    """

    def __init__(
        self,
        brake_force_max: float = 12000.0,
        latency: float = 0.2,
        margin_critical: float = 1.0,
        margin_warning: float = 4.0,
        g: float = 9.81
    ):
        self.brake_force_max = float(brake_force_max)
        self.latency = float(latency)
        self.margin_critical = float(margin_critical)
        self.margin_warning = float(margin_warning)
        self.g = float(g)
        self.latched_stage: BrakeStage = BrakeStage.OFF

    def compute_ttc(self, distance_m: float, relative_speed_kmh: float) -> float:
        v_rel_ms = relative_speed_kmh / 3.6
        if v_rel_ms > 0.05:
            return max(0.0, distance_m / v_rel_ms)
        return float("inf")

    def evaluate_state(
        self,
        vehicle_speed_kmh: float,
        mass_kg: float,
        distance_m: float,
        relative_speed_kmh: float,
        friction: float,
        road_environment: str = "Dry",
        weather_condition: str = "Clear"
    ) -> AEBDecision:
        if vehicle_speed_kmh <= 0.1 or distance_m <= 0:
            self.latched_stage = BrakeStage.OFF
            return AEBDecision(
                stage=BrakeStage.OFF,
                collision_probability=0.0,
                raw_prediction=0,
                ttc=float("inf"),
                reason="Vehicle stopped or collision occurred"
            )

        v_rel_ms = relative_speed_kmh / 3.6
        ttc = self.compute_ttc(distance_m, relative_speed_kmh)

        # Guard against opening gap or negative relative speed
        if v_rel_ms <= 0.05:
            self.latched_stage = BrakeStage.OFF
            return AEBDecision(
                stage=BrakeStage.OFF,
                collision_probability=0.0,
                raw_prediction=0,
                ttc=float("inf"),
                reason="Obstacle moving away or speed matched"
            )

        # Physics-based deceleration limit
        a_max = min(self.brake_force_max / max(1.0, mass_kg), friction * self.g)

        # Critical emergency stopping distance
        d_stop_full = (v_rel_ms * self.latency) + (v_rel_ms**2) / (2.0 * max(0.1, a_max))
        d_crit = d_stop_full + self.margin_critical

        # Warning partial stopping distance (40% effort)
        a_part = 0.4 * a_max
        d_stop_part = (v_rel_ms * self.latency) + (v_rel_ms**2) / (2.0 * max(0.1, a_part))
        d_warn = d_stop_part + self.margin_warning

        stage = BrakeStage.OFF
        reason = "Normal driving"

        if distance_m <= d_crit or ttc <= 1.4:
            stage = BrakeStage.FULL
            reason = f"Kinematic FULL: gap={distance_m:.1f}m <= d_crit={d_crit:.1f}m, TTC={ttc:.2f}s"
        elif distance_m <= d_warn or ttc <= 2.4:
            stage = BrakeStage.PARTIAL
            reason = f"Kinematic PARTIAL: gap={distance_m:.1f}m <= d_warn={d_warn:.1f}m, TTC={ttc:.2f}s"

        # Hysteresis latching
        if self.latched_stage == BrakeStage.FULL and vehicle_speed_kmh > 0.1 and relative_speed_kmh > 0.0:
            stage = BrakeStage.FULL
            reason += " (Held by hysteresis)"
        elif self.latched_stage == BrakeStage.PARTIAL and stage == BrakeStage.OFF and ttc <= 2.4:
            stage = BrakeStage.PARTIAL
            reason += " (Held by partial hysteresis)"

        self.latched_stage = stage
        surrogate_prob = min(1.0, max(0.0, d_crit / max(0.1, distance_m)))

        return AEBDecision(
            stage=stage,
            collision_probability=surrogate_prob,
            raw_prediction=1 if stage == BrakeStage.FULL else 0,
            ttc=ttc,
            reason=reason
        )

    def reset(self) -> None:
        self.latched_stage = BrakeStage.OFF


@dataclass
class CloseObstacleScenario:
    scenario_id: str
    description: str
    v_ego_kmh: float
    v_rel_kmh: float
    gap_m: float
    friction: float
    mass_kg: float
    road: str
    weather: str
    is_safe_following: bool = False  # True if scenario should NOT trigger emergency braking


def generate_close_obstacle_scenarios() -> List[CloseObstacleScenario]:
    """Generates 48 systematically varied test scenarios focusing on close-distance and low relative speed."""
    scenarios = []
    
    # 1. Critical Close-Obstacle Scenarios with Low Relative Speeds (d <= 6m, v_rel in [5, 25] km/h)
    # The identified primary failure zone of the ML model
    close_specs = [
        # (id, desc, v_ego, v_rel, gap, mu, mass, road, weather)
        ("CLOSE_01", "3.0m gap, 10 km/h closing, dry road", 30.0, 10.0, 3.0, 0.85, 1400.0, "Dry", "Clear"),
        ("CLOSE_02", "3.5m gap, 15 km/h closing, wet road", 40.0, 15.0, 3.5, 0.55, 1500.0, "Wet", "Rain"),
        ("CLOSE_03", "4.0m gap, 20 km/h closing, dry road", 45.0, 20.0, 4.0, 0.80, 1450.0, "Dry", "Clear"),
        ("CLOSE_04", "4.5m gap, 10 km/h closing, icy road", 25.0, 10.0, 4.5, 0.18, 1400.0, "Icy", "Fog"),
        ("CLOSE_05", "5.0m gap, 15 km/h closing, icy road", 30.0, 15.0, 5.0, 0.20, 1500.0, "Icy", "Clear"),
        ("CLOSE_06", "5.0m gap, 25 km/h closing, wet road", 50.0, 25.0, 5.0, 0.50, 1600.0, "Wet", "Rain"),
        ("CLOSE_07", "5.5m gap, 8 km/h closing, dry road", 25.0, 8.0, 5.5, 0.85, 1350.0, "Dry", "Clear"),
        ("CLOSE_08", "5.5m gap, 18 km/h closing, wet road", 45.0, 18.0, 5.5, 0.55, 1550.0, "Wet", "Clear"),
        ("CLOSE_09", "6.0m gap, 12 km/h closing, dry road", 35.0, 12.0, 6.0, 0.85, 1400.0, "Dry", "Clear"),
        ("CLOSE_10", "6.0m gap, 22 km/h closing, icy road", 40.0, 22.0, 6.0, 0.22, 1450.0, "Icy", "Clear"),
        ("CLOSE_11", "2.5m gap, 5 km/h creep closing, dry road", 15.0, 5.0, 2.5, 0.85, 1400.0, "Dry", "Clear"),
        ("CLOSE_12", "5.0m gap, 20 km/h closing, heavy vehicle", 40.0, 20.0, 5.0, 0.80, 1950.0, "Dry", "Clear"),
    ]
    for s in close_specs:
        scenarios.append(CloseObstacleScenario(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], False))

    # 2. Medium Distance Urban/Suburban Scenarios (d in [8, 20]m, v_rel in [20, 50] km/h)
    med_specs = [
        ("MED_01", "8.0m gap, 25 km/h closing, dry road", 40.0, 25.0, 8.0, 0.85, 1400.0, "Dry", "Clear"),
        ("MED_02", "8.0m gap, 30 km/h closing, wet road", 50.0, 30.0, 8.0, 0.55, 1500.0, "Wet", "Rain"),
        ("MED_03", "10.0m gap, 35 km/h closing, dry road", 55.0, 35.0, 10.0, 0.80, 1450.0, "Dry", "Clear"),
        ("MED_04", "10.0m gap, 20 km/h closing, icy road", 35.0, 20.0, 10.0, 0.20, 1350.0, "Icy", "Clear"),
        ("MED_05", "12.0m gap, 40 km/h closing, wet road", 60.0, 40.0, 12.0, 0.50, 1500.0, "Wet", "Fog"),
        ("MED_06", "14.0m gap, 45 km/h closing, dry road", 65.0, 45.0, 14.0, 0.85, 1400.0, "Dry", "Clear"),
        ("MED_07", "15.0m gap, 30 km/h closing, icy road", 45.0, 30.0, 15.0, 0.22, 1600.0, "Icy", "Rain"),
        ("MED_08", "16.0m gap, 50 km/h closing, wet road", 70.0, 50.0, 16.0, 0.55, 1550.0, "Wet", "Rain"),
        ("MED_09", "18.0m gap, 35 km/h closing, dry road", 60.0, 35.0, 18.0, 0.80, 1400.0, "Dry", "Clear"),
        ("MED_10", "20.0m gap, 55 km/h closing, wet road", 75.0, 55.0, 20.0, 0.50, 1650.0, "Wet", "Fog"),
        ("MED_11", "12.0m gap, 35 km/h closing, heavy vehicle", 55.0, 35.0, 12.0, 0.80, 1950.0, "Dry", "Clear"),
        ("MED_12", "15.0m gap, 25 km/h closing, light vehicle", 45.0, 25.0, 15.0, 0.85, 1100.0, "Dry", "Clear"),
    ]
    for s in med_specs:
        scenarios.append(CloseObstacleScenario(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], False))

    # 3. High-Speed Arterial and Highway Scenarios (d in [25, 55]m, v_rel in [45, 80] km/h)
    high_specs = [
        ("HIGH_01", "25.0m gap, 50 km/h closing, dry road", 70.0, 50.0, 25.0, 0.85, 1400.0, "Dry", "Clear"),
        ("HIGH_02", "25.0m gap, 60 km/h closing, wet road", 80.0, 60.0, 25.0, 0.55, 1500.0, "Wet", "Rain"),
        ("HIGH_03", "30.0m gap, 50 km/h closing, icy road", 60.0, 50.0, 30.0, 0.20, 1400.0, "Icy", "Clear"),
        ("HIGH_04", "35.0m gap, 70 km/h closing, dry road", 85.0, 70.0, 35.0, 0.85, 1500.0, "Dry", "Clear"),
        ("HIGH_05", "40.0m gap, 65 km/h closing, wet road", 80.0, 65.0, 40.0, 0.50, 1600.0, "Wet", "Fog"),
        ("HIGH_06", "45.0m gap, 80 km/h closing, dry road", 90.0, 80.0, 45.0, 0.85, 1400.0, "Dry", "Clear"),
        ("HIGH_07", "40.0m gap, 60 km/h closing, icy road", 70.0, 60.0, 40.0, 0.18, 1500.0, "Icy", "Rain"),
        ("HIGH_08", "50.0m gap, 85 km/h closing, wet road", 95.0, 85.0, 50.0, 0.55, 1650.0, "Wet", "Rain"),
        ("HIGH_09", "30.0m gap, 60 km/h closing, heavy vehicle", 80.0, 60.0, 30.0, 0.80, 1950.0, "Dry", "Clear"),
        ("HIGH_10", "35.0m gap, 45 km/h closing, light vehicle", 70.0, 45.0, 35.0, 0.85, 1100.0, "Dry", "Clear"),
        ("HIGH_11", "55.0m gap, 90 km/h closing, dry road", 95.0, 90.0, 55.0, 0.85, 1500.0, "Dry", "Clear"),
        ("HIGH_12", "30.0m gap, 40 km/h closing, wet road", 65.0, 40.0, 30.0, 0.55, 1450.0, "Wet", "Rain"),
    ]
    for s in high_specs:
        scenarios.append(CloseObstacleScenario(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], False))

    # 4. Safe Car-Following & Zero/Negative Closing Speed Scenarios (Testing False Positive / Unnecessary Braking)
    safe_specs = [
        ("SAFE_01", "30.0m gap, matched speed (0 km/h closing)", 50.0, 0.0, 30.0, 0.85, 1400.0, "Dry", "Clear"),
        ("SAFE_02", "40.0m gap, matched speed (0 km/h closing)", 70.0, 0.0, 40.0, 0.80, 1500.0, "Dry", "Clear"),
        ("SAFE_03", "50.0m gap, mild closing 5 km/h", 60.0, 5.0, 50.0, 0.85, 1400.0, "Dry", "Clear"),
        ("SAFE_04", "45.0m gap, mild closing 8 km/h, wet road", 55.0, 8.0, 45.0, 0.55, 1450.0, "Wet", "Rain"),
        ("SAFE_05", "35.0m gap, lead pulling away (-10 km/h closing)", 50.0, -10.0, 35.0, 0.85, 1400.0, "Dry", "Clear"),
        ("SAFE_06", "25.0m gap, lead pulling away (-15 km/h closing)", 60.0, -15.0, 25.0, 0.80, 1500.0, "Dry", "Clear"),
        ("SAFE_07", "40.0m gap, mild closing 10 km/h", 70.0, 10.0, 40.0, 0.85, 1400.0, "Dry", "Clear"),
        ("SAFE_08", "45.0m gap, lead pulling away (-5 km/h closing)", 40.0, -5.0, 45.0, 0.55, 1350.0, "Wet", "Clear"),
        ("SAFE_09", "50.0m gap, matched speed (0 km/h closing), icy", 40.0, 0.0, 50.0, 0.22, 1400.0, "Icy", "Fog"),
        ("SAFE_10", "30.0m gap, lead pulling away (-20 km/h closing)", 50.0, -20.0, 30.0, 0.85, 1500.0, "Dry", "Clear"),
        ("SAFE_11", "50.0m gap, mild closing 6 km/h, heavy vehicle", 65.0, 6.0, 50.0, 0.80, 1950.0, "Dry", "Clear"),
        ("SAFE_12", "60.0m gap, mild closing 10 km/h, light vehicle", 80.0, 10.0, 60.0, 0.85, 1100.0, "Dry", "Clear"),
    ]
    for s in safe_specs:
        scenarios.append(CloseObstacleScenario(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], True))

    return scenarios


@dataclass
class TriModalRecord:
    scenario_id: str
    description: str
    v_ego: float
    v_rel: float
    gap: float
    friction: float
    mass: float
    is_safe: bool
    # ML Agent Telemetry at t=0
    ml_prob_t0: float
    ml_ttc_t0: float
    ml_first_stage: str
    # Unassisted Result
    coll_no_aeb: bool
    rel_impact_no_aeb: float
    ground_impact_no_aeb: float
    # ML Agent Result
    coll_ml: bool
    rel_impact_ml: float
    ground_impact_ml: float
    prevented_ml: bool
    false_brake_ml: bool
    # Kinematic Baseline Result
    coll_kin: bool
    rel_impact_kin: float
    ground_impact_kin: float
    prevented_kin: bool
    false_brake_kin: bool


def run_tri_modal_experiment():
    print("========================================================================================")
    print("   CLOSE-OBSTACLE & KINEMATIC SAFETY BASELINE THREE-WAY BENCHMARK EVALUATION             ")
    print("========================================================================================")

    scenarios = generate_close_obstacle_scenarios()
    print(f"Total experimental scenarios: {len(scenarios)}")

    ml_agent = AEBAgent()
    kin_agent = KinematicAEBAgent()

    # Shared simulation engines
    engine_unassisted = SimulationEngine(time_step=0.01, max_duration=6.0, agent=None)
    engine_ml = SimulationEngine(time_step=0.01, max_duration=6.0, agent=ml_agent)
    engine_kin = SimulationEngine(time_step=0.01, max_duration=6.0, agent=kin_agent)

    records: List[TriModalRecord] = []

    for sc in scenarios:
        # 1. Unassisted run
        res_none = engine_unassisted.run_scenario(
            initial_speed_kmh=sc.v_ego_kmh,
            mass_kg=sc.mass_kg,
            initial_distance_m=sc.gap_m,
            relative_speed_kmh=sc.v_rel_kmh,
            friction=sc.friction,
            road_environment=sc.road,
            weather_condition=sc.weather,
            aeb_enabled=False
        )

        # 2. ML Agent run
        ml_agent.reset()
        # Probe t=0 decision
        dec_ml_t0 = ml_agent.evaluate_state(
            vehicle_speed_kmh=sc.v_ego_kmh,
            mass_kg=sc.mass_kg,
            distance_m=sc.gap_m,
            relative_speed_kmh=sc.v_rel_kmh,
            friction=sc.friction,
            road_environment=sc.road,
            weather_condition=sc.weather
        )
        ml_agent.reset()
        res_ml = engine_ml.run_scenario(
            initial_speed_kmh=sc.v_ego_kmh,
            mass_kg=sc.mass_kg,
            initial_distance_m=sc.gap_m,
            relative_speed_kmh=sc.v_rel_kmh,
            friction=sc.friction,
            road_environment=sc.road,
            weather_condition=sc.weather,
            aeb_enabled=True
        )

        # 3. Kinematic Baseline run
        kin_agent.reset()
        res_kin = engine_kin.run_scenario(
            initial_speed_kmh=sc.v_ego_kmh,
            mass_kg=sc.mass_kg,
            initial_distance_m=sc.gap_m,
            relative_speed_kmh=sc.v_rel_kmh,
            friction=sc.friction,
            road_environment=sc.road,
            weather_condition=sc.weather,
            aeb_enabled=True
        )

        # False / unnecessary braking check
        # Defined as engaging FULL emergency braking when scenario is marked safe_following (opening or zero closing)
        ml_braked = any(pt.brake_stage in ("PARTIAL", "FULL") for pt in res_ml.telemetry)
        kin_braked = any(pt.brake_stage in ("PARTIAL", "FULL") for pt in res_kin.telemetry)
        false_brake_ml = (sc.is_safe_following and ml_braked)
        false_brake_kin = (sc.is_safe_following and kin_braked)

        records.append(TriModalRecord(
            scenario_id=sc.scenario_id,
            description=sc.description,
            v_ego=sc.v_ego_kmh,
            v_rel=sc.v_rel_kmh,
            gap=sc.gap_m,
            friction=sc.friction,
            mass=sc.mass_kg,
            is_safe=sc.is_safe_following,
            ml_prob_t0=round(dec_ml_t0.collision_probability, 3),
            ml_ttc_t0=round(dec_ml_t0.ttc, 2) if dec_ml_t0.ttc != float("inf") else 999.0,
            ml_first_stage=dec_ml_t0.stage.name,
            coll_no_aeb=res_none.collision_occurred,
            rel_impact_no_aeb=round(res_none.impact_relative_speed_kmh, 2),
            ground_impact_no_aeb=round(res_none.impact_speed_kmh, 2),
            coll_ml=res_ml.collision_occurred,
            rel_impact_ml=round(res_ml.impact_relative_speed_kmh, 2),
            ground_impact_ml=round(res_ml.impact_speed_kmh, 2),
            prevented_ml=(res_none.collision_occurred and not res_ml.collision_occurred),
            false_brake_ml=false_brake_ml,
            coll_kin=res_kin.collision_occurred,
            rel_impact_kin=round(res_kin.impact_relative_speed_kmh, 2),
            ground_impact_kin=round(res_kin.impact_speed_kmh, 2),
            prevented_kin=(res_none.collision_occurred and not res_kin.collision_occurred),
            false_brake_kin=false_brake_kin
        ))

    return records


def print_experiment_results(records: List[TriModalRecord]):
    total = len(records)
    hazard_scenarios = [r for r in records if not r.is_safe]
    safe_scenarios = [r for r in records if r.is_safe]

    # Metrics computation
    c_none = sum(1 for r in records if r.coll_no_aeb)
    c_ml = sum(1 for r in records if r.coll_ml)
    c_kin = sum(1 for r in records if r.coll_kin)

    prev_ml = sum(1 for r in records if r.prevented_ml)
    prev_kin = sum(1 for r in records if r.prevented_kin)

    avg_rel_none = sum(r.rel_impact_no_aeb for r in records if r.coll_no_aeb) / c_none if c_none > 0 else 0.0
    avg_rel_ml = sum(r.rel_impact_ml for r in records if r.coll_ml) / c_ml if c_ml > 0 else 0.0
    avg_rel_kin = sum(r.rel_impact_kin for r in records if r.coll_kin) / c_kin if c_kin > 0 else 0.0

    max_rel_none = max([r.rel_impact_no_aeb for r in records if r.coll_no_aeb] + [0.0])
    max_rel_ml = max([r.rel_impact_ml for r in records if r.coll_ml] + [0.0])
    max_rel_kin = max([r.rel_impact_kin for r in records if r.coll_kin] + [0.0])

    fb_ml = sum(1 for r in safe_scenarios if r.false_brake_ml)
    fb_kin = sum(1 for r in safe_scenarios if r.false_brake_kin)

    print("\n------------------------------ SCENARIO-BY-SCENARIO SUMMARY ------------------------------")
    print(f"{'ID':<9} {'Gap':<5} {'v_rel':<6} {'mu':<5} {'ML P0':<6} {'No AEB (Rel)':<13} {'ML AEB (Rel)':<13} {'Kin AEB (Rel)':<14} {'ML Prev?':<9} {'Kin Prev?'}")
    print("-" * 98)
    for r in records:
        no_str = f"{r.rel_impact_no_aeb:.1f} km/h" if r.coll_no_aeb else "Safe"
        ml_str = f"{r.rel_impact_ml:.1f} km/h" if r.coll_ml else "Safe"
        kin_str = f"{r.rel_impact_kin:.1f} km/h" if r.coll_kin else "Safe"
        p_ml_str = "YES" if r.prevented_ml else ("N/A" if not r.coll_no_aeb else "NO")
        p_kin_str = "YES" if r.prevented_kin else ("N/A" if not r.coll_no_aeb else "NO")
        print(f"{r.scenario_id:<9} {r.gap:<5.1f} {r.v_rel:<6.1f} {r.friction:<5.2f} {r.ml_prob_t0:<6.2f} {no_str:<13} {ml_str:<13} {kin_str:<14} {p_ml_str:<9} {p_kin_str}")

    print("\n============================== AGGREGATE COMPARISON ==============================")
    print(f"Total Scenarios Evaluated            : {total} (Hazard: {len(hazard_scenarios)}, Safe Baseline: {len(safe_scenarios)})")
    print(f"Collisions without AEB (Baseline)    : {c_none} / {total} ({c_none/total*100:.1f}%)")
    print(f"Collisions with ML-based AEB         : {c_ml} / {total} ({c_ml/total*100:.1f}%) [Prevented: {prev_ml}]")
    print(f"Collisions with Kinematic Safety AEB : {c_kin} / {total} ({c_kin/total*100:.1f}%) [Prevented: {prev_kin}]")
    print(f"Relative Collision Reduction (ML)    : {(c_none - c_ml)/c_none*100:.2f}%")
    print(f"Relative Collision Reduction (Kin)   : {(c_none - c_kin)/c_none*100:.2f}%")
    print("----------------------------------------------------------------------------------")
    print(f"Avg Closing Impact Speed (Collisions): No AEB: {avg_rel_none:.2f} km/h | ML: {avg_rel_ml:.2f} km/h | Kin: {avg_rel_kin:.2f} km/h")
    print(f"Max Closing Impact Speed (Collisions): No AEB: {max_rel_none:.2f} km/h | ML: {max_rel_ml:.2f} km/h | Kin: {max_rel_kin:.2f} km/h")
    print(f"False / Unnecessary Braking on Safe  : ML: {fb_ml}/{len(safe_scenarios)} | Kinematic: {fb_kin}/{len(safe_scenarios)}")
    print("==================================================================================\n")

    # Detailed Grouping by Relative Speed and Distance
    print("--- Performance in Critical Close-Obstacle Zone (d <= 6m, 12 scenarios) ---")
    close_records = [r for r in records if r.gap <= 6.0 and not r.is_safe]
    c_close_none = sum(1 for r in close_records if r.coll_no_aeb)
    c_close_ml = sum(1 for r in close_records if r.coll_ml)
    c_close_kin = sum(1 for r in close_records if r.coll_kin)
    print(f"  Unassisted Collisions   : {c_close_none} / {len(close_records)}")
    print(f"  ML-Agent Collisions     : {c_close_ml} / {len(close_records)} (Prevented: {c_close_none - c_close_ml})")
    print(f"  Kinematic-AEB Collisions: {c_close_kin} / {len(close_records)} (Prevented: {c_close_none - c_close_kin})")

    # Grouping by Friction
    print("\n--- Performance Grouped by Surface Friction ---")
    for mu_label, mu_cond in [("Icy (mu <= 0.25)", lambda m: m <= 0.25),
                              ("Wet (0.25 < mu <= 0.65)", lambda m: 0.25 < m <= 0.65),
                              ("Dry (mu > 0.65)", lambda m: m > 0.65)]:
        sub = [r for r in records if mu_cond(r.friction) and not r.is_safe]
        cn = sum(1 for r in sub if r.coll_no_aeb)
        cm = sum(1 for r in sub if r.coll_ml)
        ck = sum(1 for r in sub if r.coll_kin)
        print(f"  {mu_label:<25}: Scenarios={len(sub):<2} | No AEB={cn:<2} | ML AEB={cm:<2} (Prev {cn-cm}) | Kin AEB={ck:<2} (Prev {cn-ck})")


if __name__ == "__main__":
    recs = run_tri_modal_experiment()
    print_experiment_results(recs)
