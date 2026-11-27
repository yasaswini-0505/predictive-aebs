"""
Expanded 500-Scenario Systematic AEB Benchmark Experiment.

Evaluates 500 diverse, reproducible driving scenarios across 3 configurations:
1. Unassisted vehicle (No AEB)
2. Existing ML-based AEB agent (Random Forest)
3. Physically justified Kinematic Safety Baseline (KinematicAEBAgent)

Covers:
- Close obstacles (d <= 6m, low relative speed)
- Medium distance urban/suburban traffic
- High-speed arterial/highway approaches
- Physically unavoidable collision scenarios (testing impact mitigation)
- Safe-following and opening-gap scenarios (testing unnecessary / false braking)

Does not modify any production files or existing saved models.
Uses fixed random seed (42) for exact reproducibility.
"""

import sys
import os
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulation_engine import SimulationEngine, SimulationOutcome, SimulationResult
from braking_controller import BrakingController, BrakeStage
from aeb_agent import AEBAgent, AEBDecision
from experiment_close_obstacle_baseline import KinematicAEBAgent


class FastMLAEBAgent(AEBAgent):
    """
    Inference-optimized wrapper for AEBAgent.
    Bypasses pandas DataFrame reconstruction in tight simulation loops while
    producing bit-exact predictions from the saved RandomForestClassifier pipeline.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clf = self.model.named_steps['classifier']
        self.road_map = {"Dry": 0, "Icy": 1, "Wet": 2}
        self.weather_map = {"Clear": 0, "Fog": 1, "Rain": 2}
        self._buf = np.zeros((1, 11), dtype=np.float64)

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

        ttc = self.compute_ttc(distance_m, relative_speed_kmh)

        # Populate buffer: 6 one-hot + 5 numerical features
        self._buf.fill(0.0)
        r_idx = self.road_map.get(road_environment, 0)
        w_idx = self.weather_map.get(weather_condition, 0)
        self._buf[0, r_idx] = 1.0
        self._buf[0, 3 + w_idx] = 1.0
        self._buf[0, 6] = float(vehicle_speed_kmh)
        self._buf[0, 7] = float(mass_kg)
        self._buf[0, 8] = max(0.1, float(distance_m))
        self._buf[0, 9] = max(0.0, float(relative_speed_kmh))
        self._buf[0, 10] = float(friction)

        prob_classes = self.clf.predict_proba(self._buf)[0]
        prob_collision = float(prob_classes[1]) if len(prob_classes) > 1 else float(prob_classes[0])
        raw_pred = 1 if prob_collision >= 0.5 else 0

        stage = BrakeStage.OFF
        reason = "Normal driving"

        if prob_collision >= self.full_risk_threshold or (ttc <= self.ttc_critical and prob_collision >= 0.20):
            stage = BrakeStage.FULL
            reason = f"Full emergency braking: P(collision)={prob_collision:.2f}, TTC={ttc:.2f}s"
        elif prob_collision >= self.partial_risk_threshold or (ttc <= self.ttc_warning and prob_collision >= 0.15):
            stage = BrakeStage.PARTIAL
            reason = f"Partial warning braking: P(collision)={prob_collision:.2f}, TTC={ttc:.2f}s"

        if self.hysteresis_hold:
            if self.latched_stage == BrakeStage.FULL and vehicle_speed_kmh > 0.1 and relative_speed_kmh > 0.0:
                stage = BrakeStage.FULL
                reason += " (Held by AEB hysteresis)"
            elif self.latched_stage == BrakeStage.PARTIAL and stage == BrakeStage.OFF and ttc <= self.ttc_warning:
                stage = BrakeStage.PARTIAL
                reason += " (Held by partial hysteresis)"

        self.latched_stage = stage

        return AEBDecision(
            stage=stage,
            collision_probability=prob_collision,
            raw_prediction=raw_pred,
            ttc=ttc,
            reason=reason
        )


@dataclass
class ScenarioSpec:
    scenario_id: str
    cohort: str
    v_ego_kmh: float
    v_rel_kmh: float
    gap_m: float
    friction: float
    mass_kg: float
    road: str
    weather: str
    is_safe_following: bool = False
    is_physically_unavoidable: bool = False


@dataclass
class ScenarioEvaluationResult:
    scenario_id: str
    cohort: str
    v_ego: float
    v_rel: float
    v_obs: float
    gap: float
    friction: float
    mass: float
    road: str
    weather: str
    is_safe: bool
    is_impossible: bool
    # ML initial assessment
    ml_prob_t0: float
    ml_ttc_t0: float
    ml_stage_t0: str
    # Unassisted run
    coll_no_aeb: bool
    rel_impact_no_aeb: float
    ground_impact_no_aeb: float
    # ML AEB run
    coll_ml: bool
    rel_impact_ml: float
    ground_impact_ml: float
    prevented_ml: bool
    unnecessary_brake_ml: bool
    # Kinematic AEB run
    coll_kin: bool
    rel_impact_kin: float
    ground_impact_kin: float
    prevented_kin: bool
    unnecessary_brake_kin: bool


def generate_500_scenarios(seed: int = 42) -> List[ScenarioSpec]:
    """
    Generates 500 diverse, reproducible evaluation scenarios across 5 distinct cohorts:
    1. Close Obstacle (100) - d in [2.0, 6.0]m, v_rel in [5, 25] km/h
    2. Medium Distance (100) - d in (6.0, 20.0]m, v_rel in [20, 55] km/h
    3. High Speed (100) - d in (20.0, 60.0]m, v_rel in [50, 95] km/h
    4. Physically Impossible to Avoid (100) - d < required physics stopping distance
    5. Safe Following / Nuisance Test (100) - Matched, opening, or large headway
    """
    rng = np.random.default_rng(seed)
    scenarios: List[ScenarioSpec] = []

    def get_env_and_weather(mu: float):
        if mu > 0.65:
            return "Dry", rng.choice(["Clear", "Clear", "Fog"])
        elif mu > 0.28:
            return "Wet", rng.choice(["Rain", "Rain", "Clear"])
        else:
            return "Icy", rng.choice(["Fog", "Rain", "Clear"])

    # -------------------------------------------------------------
    # Cohort 1: Critical Close-Obstacle Zone (100 scenarios)
    # -------------------------------------------------------------
    for i in range(100):
        gap = round(float(rng.uniform(2.0, 6.0)), 2)
        v_rel = round(float(rng.uniform(5.0, 25.0)), 1)
        # 50% stationary obstacle (v_obs = 0 -> v_ego = v_rel), 50% moving lead vehicle
        if rng.random() < 0.5:
            v_ego = v_rel
        else:
            v_ego = round(float(rng.uniform(v_rel + 10.0, v_rel + 45.0)), 1)
        
        mu = round(float(rng.choice([0.85, 0.80, 0.75, 0.55, 0.50, 0.45, 0.22, 0.18])), 2)
        mass = round(float(rng.uniform(1100.0, 1950.0)), 1)
        road, weather = get_env_and_weather(mu)

        scenarios.append(ScenarioSpec(
            scenario_id=f"COH1_CLOSE_{i+1:03d}",
            cohort="Close Obstacle (d <= 6m)",
            v_ego_kmh=v_ego,
            v_rel_kmh=v_rel,
            gap_m=gap,
            friction=mu,
            mass_kg=mass,
            road=road,
            weather=weather,
            is_safe_following=False,
            is_physically_unavoidable=False
        ))

    # -------------------------------------------------------------
    # Cohort 2: Medium Distance Urban/Suburban (100 scenarios)
    # -------------------------------------------------------------
    for i in range(100):
        gap = round(float(rng.uniform(6.5, 20.0)), 2)
        v_rel = round(float(rng.uniform(20.0, 55.0)), 1)
        if rng.random() < 0.35:
            v_ego = v_rel
        else:
            v_ego = round(float(rng.uniform(v_rel + 15.0, v_rel + 50.0)), 1)
        
        mu = round(float(rng.choice([0.88, 0.82, 0.78, 0.60, 0.52, 0.40, 0.25, 0.19])), 2)
        mass = round(float(rng.uniform(1150.0, 1900.0)), 1)
        road, weather = get_env_and_weather(mu)

        scenarios.append(ScenarioSpec(
            scenario_id=f"COH2_MED_{i+1:03d}",
            cohort="Medium Distance (6m < d <= 20m)",
            v_ego_kmh=v_ego,
            v_rel_kmh=v_rel,
            gap_m=gap,
            friction=mu,
            mass_kg=mass,
            road=road,
            weather=weather,
            is_safe_following=False,
            is_physically_unavoidable=False
        ))

    # -------------------------------------------------------------
    # Cohort 3: High-Speed Arterial/Highway (100 scenarios)
    # -------------------------------------------------------------
    for i in range(100):
        gap = round(float(rng.uniform(20.5, 60.0)), 2)
        v_rel = round(float(rng.uniform(50.0, 95.0)), 1)
        if rng.random() < 0.20:
            v_ego = min(100.0, v_rel)
        else:
            v_ego = min(110.0, round(float(rng.uniform(v_rel + 10.0, v_rel + 35.0)), 1))
        
        mu = round(float(rng.choice([0.85, 0.80, 0.65, 0.55, 0.45, 0.22, 0.18])), 2)
        mass = round(float(rng.uniform(1200.0, 1950.0)), 1)
        road, weather = get_env_and_weather(mu)

        scenarios.append(ScenarioSpec(
            scenario_id=f"COH3_HIGH_{i+1:03d}",
            cohort="High Speed (20m < d <= 60m)",
            v_ego_kmh=v_ego,
            v_rel_kmh=v_rel,
            gap_m=gap,
            friction=mu,
            mass_kg=mass,
            road=road,
            weather=weather,
            is_safe_following=False,
            is_physically_unavoidable=False
        ))

    # -------------------------------------------------------------
    # Cohort 4: Physically Unavoidable Scenarios (100 scenarios)
    # Gap is strictly less than physical minimum stopping distance:
    # d < (v_rel * tau) + v_rel^2 / (2 * a_max)
    # -------------------------------------------------------------
    for i in range(100):
        v_rel = round(float(rng.uniform(30.0, 80.0)), 1)
        v_ego = round(float(rng.uniform(v_rel, v_rel + 30.0)), 1)
        mu = round(float(rng.choice([0.15, 0.20, 0.25, 0.40, 0.50, 0.70])), 2)
        mass = round(float(rng.uniform(1200.0, 1850.0)), 1)
        
        # Calculate strict physical stopping distance with zero margin
        v_rel_ms = v_rel / 3.6
        a_max = min(12000.0 / mass, mu * 9.81)
        d_req = (v_rel_ms * 0.20) + (v_rel_ms**2) / (2.0 * a_max)
        
        # Make gap 30% to 75% of required distance -> mathematically cannot avoid collision
        fraction = float(rng.uniform(0.30, 0.75))
        gap = max(1.5, round(d_req * fraction, 2))
        road, weather = get_env_and_weather(mu)

        scenarios.append(ScenarioSpec(
            scenario_id=f"COH4_UNAVOID_{i+1:03d}",
            cohort="Physically Unavoidable (Mitigation)",
            v_ego_kmh=v_ego,
            v_rel_kmh=v_rel,
            gap_m=gap,
            friction=mu,
            mass_kg=mass,
            road=road,
            weather=weather,
            is_safe_following=False,
            is_physically_unavoidable=True
        ))

    # -------------------------------------------------------------
    # Cohort 5: Safe Following / Nuisance Test (100 scenarios)
    # -------------------------------------------------------------
    for i in range(100):
        mode = rng.choice(["matched", "opening", "large_headway"])
        v_ego = round(float(rng.uniform(40.0, 90.0)), 1)
        mu = round(float(rng.choice([0.85, 0.80, 0.55, 0.22])), 2)
        mass = round(float(rng.uniform(1200.0, 1800.0)), 1)
        road, weather = get_env_and_weather(mu)

        if mode == "matched":
            # Speed matched: v_rel = 0
            v_rel = 0.0
            gap = round(float(rng.uniform(25.0, 65.0)), 2)
        elif mode == "opening":
            # Lead vehicle pulling away: v_rel < 0
            v_rel = round(float(rng.uniform(-25.0, -3.0)), 1)
            gap = round(float(rng.uniform(18.0, 50.0)), 2)
        else:
            # Large headway, very mild closing speed (TTC > 15s)
            v_rel = round(float(rng.uniform(2.0, 8.0)), 1)
            gap = round(float(rng.uniform(45.0, 85.0)), 2)

        scenarios.append(ScenarioSpec(
            scenario_id=f"COH5_SAFE_{i+1:03d}",
            cohort="Safe Following / Nuisance Test",
            v_ego_kmh=v_ego,
            v_rel_kmh=v_rel,
            gap_m=gap,
            friction=mu,
            mass_kg=mass,
            road=road,
            weather=weather,
            is_safe_following=True,
            is_physically_unavoidable=False
        ))

    return scenarios


_PROCESS_ML_AGENT = None
_PROCESS_KIN_AGENT = None
_PROCESS_ENGINES = None


def get_process_resources():
    global _PROCESS_ML_AGENT, _PROCESS_KIN_AGENT, _PROCESS_ENGINES
    if _PROCESS_ML_AGENT is None:
        _PROCESS_ML_AGENT = FastMLAEBAgent()
        _PROCESS_KIN_AGENT = KinematicAEBAgent()
        _PROCESS_ENGINES = (
            SimulationEngine(time_step=0.01, max_duration=5.0, agent=None),
            SimulationEngine(time_step=0.01, max_duration=5.0, agent=_PROCESS_ML_AGENT),
            SimulationEngine(time_step=0.01, max_duration=5.0, agent=_PROCESS_KIN_AGENT),
        )
    return _PROCESS_ML_AGENT, _PROCESS_KIN_AGENT, _PROCESS_ENGINES


def evaluate_scenario_worker(spec: ScenarioSpec) -> ScenarioEvaluationResult:
    """Evaluates a single scenario under all three configurations with fresh agent states."""
    ml_agent, kin_agent, (engine_none, engine_ml, engine_kin) = get_process_resources()


    # 1. Probe ML Agent at t=0
    ml_agent.reset()
    dec_t0 = ml_agent.evaluate_state(
        vehicle_speed_kmh=spec.v_ego_kmh,
        mass_kg=spec.mass_kg,
        distance_m=spec.gap_m,
        relative_speed_kmh=spec.v_rel_kmh,
        friction=spec.friction,
        road_environment=spec.road,
        weather_condition=spec.weather
    )

    # 2. Run unassisted (No AEB)
    res_none = engine_none.run_scenario(
        initial_speed_kmh=spec.v_ego_kmh,
        mass_kg=spec.mass_kg,
        initial_distance_m=spec.gap_m,
        relative_speed_kmh=spec.v_rel_kmh,
        friction=spec.friction,
        road_environment=spec.road,
        weather_condition=spec.weather,
        aeb_enabled=False
    )

    # 3. Run with ML AEB
    ml_agent.reset()
    res_ml = engine_ml.run_scenario(
        initial_speed_kmh=spec.v_ego_kmh,
        mass_kg=spec.mass_kg,
        initial_distance_m=spec.gap_m,
        relative_speed_kmh=spec.v_rel_kmh,
        friction=spec.friction,
        road_environment=spec.road,
        weather_condition=spec.weather,
        aeb_enabled=True
    )

    # 4. Run with Kinematic AEB
    kin_agent.reset()
    res_kin = engine_kin.run_scenario(
        initial_speed_kmh=spec.v_ego_kmh,
        mass_kg=spec.mass_kg,
        initial_distance_m=spec.gap_m,
        relative_speed_kmh=spec.v_rel_kmh,
        friction=spec.friction,
        road_environment=spec.road,
        weather_condition=spec.weather,
        aeb_enabled=True
    )

    # Check unnecessary braking on safe-following scenarios
    ml_braked = any(pt.brake_stage in ("PARTIAL", "FULL") for pt in res_ml.telemetry)
    kin_braked = any(pt.brake_stage in ("PARTIAL", "FULL") for pt in res_kin.telemetry)
    unnecessary_ml = bool(spec.is_safe_following and ml_braked)
    unnecessary_kin = bool(spec.is_safe_following and kin_braked)

    v_obs_kmh = max(0.0, round(spec.v_ego_kmh - spec.v_rel_kmh, 2))

    return ScenarioEvaluationResult(
        scenario_id=spec.scenario_id,
        cohort=spec.cohort,
        v_ego=spec.v_ego_kmh,
        v_rel=spec.v_rel_kmh,
        v_obs=v_obs_kmh,
        gap=spec.gap_m,
        friction=spec.friction,
        mass=spec.mass_kg,
        road=spec.road,
        weather=spec.weather,
        is_safe=spec.is_safe_following,
        is_impossible=spec.is_physically_unavoidable,
        ml_prob_t0=round(dec_t0.collision_probability, 3),
        ml_ttc_t0=round(dec_t0.ttc, 2) if dec_t0.ttc != float("inf") else 999.0,
        ml_stage_t0=dec_t0.stage.name,
        coll_no_aeb=res_none.collision_occurred,
        rel_impact_no_aeb=round(res_none.impact_relative_speed_kmh, 2),
        ground_impact_no_aeb=round(res_none.impact_speed_kmh, 2),
        coll_ml=res_ml.collision_occurred,
        rel_impact_ml=round(res_ml.impact_relative_speed_kmh, 2),
        ground_impact_ml=round(res_ml.impact_speed_kmh, 2),
        prevented_ml=bool(res_none.collision_occurred and not res_ml.collision_occurred),
        unnecessary_brake_ml=unnecessary_ml,
        coll_kin=res_kin.collision_occurred,
        rel_impact_kin=round(res_kin.impact_relative_speed_kmh, 2),
        ground_impact_kin=round(res_kin.impact_speed_kmh, 2),
        prevented_kin=bool(res_none.collision_occurred and not res_kin.collision_occurred),
        unnecessary_brake_kin=unnecessary_kin
    )


def run_expanded_benchmark(n_workers: int = 8) -> List[ScenarioEvaluationResult]:
    scenarios = generate_500_scenarios(seed=42)
    print(f"Executing 500-scenario expanded benchmark using {n_workers} parallel workers...")
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(evaluate_scenario_worker, scenarios))

    elapsed = time.time() - t0
    print(f"Completed 500 scenarios in {elapsed:.2f} seconds ({elapsed/500*1000:.1f} ms/scenario).")
    return results


def print_detailed_report(results: List[ScenarioEvaluationResult]):
    total = len(results)
    c_none = sum(1 for r in results if r.coll_no_aeb)
    c_ml = sum(1 for r in results if r.coll_ml)
    c_kin = sum(1 for r in results if r.coll_kin)

    prev_ml = sum(1 for r in results if r.prevented_ml)
    prev_kin = sum(1 for r in results if r.prevented_kin)

    # Closing impact speed among collisions ONLY
    avg_rel_none = sum(r.rel_impact_no_aeb for r in results if r.coll_no_aeb) / c_none if c_none > 0 else 0.0
    avg_rel_ml = sum(r.rel_impact_ml for r in results if r.coll_ml) / c_ml if c_ml > 0 else 0.0
    avg_rel_kin = sum(r.rel_impact_kin for r in results if r.coll_kin) / c_kin if c_kin > 0 else 0.0

    max_rel_none = max([r.rel_impact_no_aeb for r in results if r.coll_no_aeb] + [0.0])
    max_rel_ml = max([r.rel_impact_ml for r in results if r.coll_ml] + [0.0])
    max_rel_kin = max([r.rel_impact_kin for r in results if r.coll_kin] + [0.0])

    safe_subset = [r for r in results if r.is_safe]
    unnecessary_ml = sum(1 for r in safe_subset if r.unnecessary_brake_ml)
    unnecessary_kin = sum(1 for r in safe_subset if r.unnecessary_brake_kin)

    print("\n" + "=" * 95)
    print("           EXPANDED 500-SCENARIO AEB BENCHMARK: EXECUTIVE SUMMARY REPORT           ")
    print("=" * 95)
    print(f"Total Scenarios Evaluated            : {total}")
    print(f"Collision Count without AEB          : {c_none} / {total} ({c_none/total*100:.2f}%)")
    print(f"Collision Count with ML-based AEB    : {c_ml} / {total} ({c_ml/total*100:.2f}%) [Prevented: {prev_ml}]")
    print(f"Collision Count with Kinematic AEB   : {c_kin} / {total} ({c_kin/total*100:.2f}%) [Prevented: {prev_kin}]")
    print(f"Relative Collision Reduction (ML)    : {(c_none - c_ml)/c_none*100:.2f}%")
    print(f"Relative Collision Reduction (Kin)   : {(c_none - c_kin)/c_none*100:.2f}%")
    print("-" * 95)
    print("Closing Impact Speed (Collisions Only):")
    print(f"  Unassisted Baseline (No AEB)       : Avg = {avg_rel_none:.2f} km/h | Max = {max_rel_none:.2f} km/h")
    print(f"  ML-based AEB Agent                 : Avg = {avg_rel_ml:.2f} km/h | Max = {max_rel_ml:.2f} km/h")
    print(f"  Kinematic Safety Baseline          : Avg = {avg_rel_kin:.2f} km/h | Max = {max_rel_kin:.2f} km/h")
    print("-" * 95)
    print(f"Unnecessary / False Braking on Safe  : ML = {unnecessary_ml}/{len(safe_subset)} ({unnecessary_ml/len(safe_subset)*100:.1f}%) | Kin = {unnecessary_kin}/{len(safe_subset)} ({unnecessary_kin/len(safe_subset)*100:.1f}%)")
    print("=" * 95)

    # Cohort breakdown
    cohort_names = [
        "Close Obstacle (d <= 6m)",
        "Medium Distance (6m < d <= 20m)",
        "High Speed (20m < d <= 60m)",
        "Physically Unavoidable (Mitigation)",
        "Safe Following / Nuisance Test"
    ]
    print("\n--- RESULTS GROUPED BY TESTING COHORT (100 SCENARIOS EACH) ---")
    print(f"{'Cohort Name':<36} {'No AEB':<10} {'ML AEB':<12} {'ML Prev':<10} {'Kin AEB':<12} {'Kin Prev'}")
    print("-" * 92)
    for c_name in cohort_names:
        sub = [r for r in results if r.cohort == c_name]
        cn = sum(1 for r in sub if r.coll_no_aeb)
        cm = sum(1 for r in sub if r.coll_ml)
        ck = sum(1 for r in sub if r.coll_kin)
        pm = sum(1 for r in sub if r.prevented_ml)
        pk = sum(1 for r in sub if r.prevented_kin)
        print(f"{c_name:<36} {cn:<10} {cm:<12} {pm:<10} {ck:<12} {pk}")

    # Friction breakdown on hazardous scenarios (excluding safe following)
    hazard_results = [r for r in results if not r.is_safe]
    print("\n--- RESULTS GROUPED BY ROAD SURFACE FRICTION (400 HAZARD SCENARIOS) ---")
    print(f"{'Friction Regime':<25} {'Scenarios':<11} {'No AEB':<10} {'ML AEB':<12} {'ML Prev':<10} {'Kin AEB':<12} {'Kin Prev'}")
    print("-" * 92)
    for f_label, cond in [
        ("Icy (mu <= 0.25)", lambda m: m <= 0.25),
        ("Wet (0.25 < mu <= 0.65)", lambda m: 0.25 < m <= 0.65),
        ("Dry (mu > 0.65)", lambda m: m > 0.65)
    ]:
        sub = [r for r in hazard_results if cond(r.friction)]
        cn = sum(1 for r in sub if r.coll_no_aeb)
        cm = sum(1 for r in sub if r.coll_ml)
        ck = sum(1 for r in sub if r.coll_kin)
        pm = sum(1 for r in sub if r.prevented_ml)
        pk = sum(1 for r in sub if r.prevented_kin)
        print(f"{f_label:<25} {len(sub):<11} {cn:<10} {cm:<12} {pm:<10} {ck:<12} {pk}")

    # Impact mitigation in unavoidable crashes
    unavoidable = [r for r in results if r.is_impossible]
    avg_rel_unav_none = sum(r.rel_impact_no_aeb for r in unavoidable) / len(unavoidable)
    avg_rel_unav_ml = sum(r.rel_impact_ml for r in unavoidable) / len(unavoidable)
    avg_rel_unav_kin = sum(r.rel_impact_kin for r in unavoidable) / len(unavoidable)
    print("\n--- IMPACT MITIGATION IN PHYSICALLY UNAVOIDABLE CRASHES (100 SCENARIOS) ---")
    print(f"Unassisted Baseline Impact Closing Speed : {avg_rel_unav_none:.2f} km/h")
    print(f"ML AEB Agent Impact Closing Speed        : {avg_rel_unav_ml:.2f} km/h (Reduction: {avg_rel_unav_none - avg_rel_unav_ml:.2f} km/h)")
    print(f"Kinematic AEB Impact Closing Speed       : {avg_rel_unav_kin:.2f} km/h (Reduction: {avg_rel_unav_none - avg_rel_unav_kin:.2f} km/h)")
    print("=" * 95 + "\n")


def save_results_to_csv(results: List[ScenarioEvaluationResult], filepath: str):
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(filepath, index=False)
    print(f"Detailed results saved to: {filepath}")


if __name__ == "__main__":
    results = run_expanded_benchmark(n_workers=8)
    print_detailed_report(results)
    out_csv = str(PROJECT_ROOT / "aeb_expanded_500_results.csv")
    save_results_to_csv(results, out_csv)
