"""
Four-Way Comparative Evaluation of Dynamic Lead-Vehicle Braking Scenarios.

Compares four AEB configurations:
1. Unassisted Baseline (No AEB)
2. ML-Only AEB (Random Forest without physics fallback)
3. Physics-Only AEB (Kinematic Stopping Distance)
4. Hybrid AEB (Random Forest + Physics-Based Safety Fallback)

Evaluates dynamic lead-vehicle deceleration profiles:
- Constant cruising (0.0 m/s^2)
- Gradual lead slowing (-2.0 m/s^2)
- Sudden hard braking (-5.0 to -6.0 m/s^2)
- Emergency maximum braking (-8.0 m/s^2)
- Low-friction (wet/icy) dynamic braking
- Heavy ego vehicle dynamic approach
- Safe headway following

All configurations are evaluated under strictly identical initial conditions.
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

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_simulation_engine import DynamicSimulationEngine, DynamicSimulationResult
from evaluate_hybrid_500_comparison import FastComparativeAEBAgent
from experiment_close_obstacle_baseline import KinematicAEBAgent
from simulation_engine import SimulationOutcome
from braking_controller import BrakeStage


@dataclass
class DynamicScenarioSpec:
    scenario_id: str
    category: str
    description: str
    v_ego_kmh: float
    v_lead_init_kmh: float
    gap_m: float
    friction: float
    mass_kg: float
    road: str
    weather: str
    lead_accel_ms2: float
    lead_brake_start_s: float
    lead_target_kmh: float
    is_safe_following: bool = False


@dataclass
class DynamicComparisonRecord:
    scenario_id: str
    category: str
    description: str
    v_ego_kmh: float
    v_lead_kmh: float
    gap_m: float
    friction: float
    mass_kg: float
    lead_accel_ms2: float
    is_safe: bool
    # 1. No AEB
    coll_none: bool
    min_gap_none: float
    impact_speed_none: float
    rel_impact_none: float
    # 2. ML Only
    coll_ml: bool
    min_gap_ml: float
    impact_speed_ml: float
    rel_impact_ml: float
    t_brake_ml: Optional[float]
    prevented_ml: bool
    unnecessary_ml: bool
    # 3. Kinematic
    coll_kin: bool
    min_gap_kin: float
    impact_speed_kin: float
    rel_impact_kin: float
    t_brake_kin: Optional[float]
    prevented_kin: bool
    unnecessary_kin: bool
    # 4. Hybrid
    coll_hyb: bool
    min_gap_hyb: float
    impact_speed_hyb: float
    rel_impact_hyb: float
    t_brake_hyb: Optional[float]
    prevented_hyb: bool
    unnecessary_hyb: bool


def generate_representative_dynamic_scenarios() -> List[DynamicScenarioSpec]:
    """Generates the targeted matrix of dynamic lead braking scenarios."""
    return [
        DynamicScenarioSpec(
            scenario_id="DYN_01_CONST",
            category="Constant Speed Cruising",
            description="Equal speed car-following on dry road, lead maintains 60 km/h",
            v_ego_kmh=60.0, v_lead_init_kmh=60.0, gap_m=25.0, friction=0.85, mass_kg=1500.0,
            road="Dry", weather="Clear", lead_accel_ms2=0.0, lead_brake_start_s=0.0,
            lead_target_kmh=60.0, is_safe_following=True
        ),
        DynamicScenarioSpec(
            scenario_id="DYN_02_GRADUAL",
            category="Gradual Lead Slowing",
            description="Lead vehicle gradually slows from 60 to 20 km/h at -2.0 m/s^2",
            v_ego_kmh=60.0, v_lead_init_kmh=60.0, gap_m=30.0, friction=0.80, mass_kg=1500.0,
            road="Dry", weather="Clear", lead_accel_ms2=-2.0, lead_brake_start_s=0.5,
            lead_target_kmh=20.0, is_safe_following=False
        ),
        DynamicScenarioSpec(
            scenario_id="DYN_03_MODERATE_HWY",
            category="Moderate Highway Braking",
            description="Highway cruising 80 km/h, lead brakes moderately at -4.0 m/s^2 to full stop",
            v_ego_kmh=80.0, v_lead_init_kmh=80.0, gap_m=35.0, friction=0.85, mass_kg=1500.0,
            road="Dry", weather="Clear", lead_accel_ms2=-4.0, lead_brake_start_s=0.5,
            lead_target_kmh=0.0, is_safe_following=False
        ),
        DynamicScenarioSpec(
            scenario_id="DYN_04_SUDDEN_BRAKE",
            category="Sudden Hard Braking",
            description="Urban/arterial 50 km/h, lead vehicle hard brakes at -6.0 m/s^2 to full stop",
            v_ego_kmh=50.0, v_lead_init_kmh=50.0, gap_m=20.0, friction=0.85, mass_kg=1500.0,
            road="Dry", weather="Clear", lead_accel_ms2=-6.0, lead_brake_start_s=0.4,
            lead_target_kmh=0.0, is_safe_following=False
        ),
        DynamicScenarioSpec(
            scenario_id="DYN_05_EMERGENCY_MAX",
            category="Emergency Maximum Braking",
            description="Aggressive lead emergency stop at -8.0 m/s^2 from 60 km/h",
            v_ego_kmh=60.0, v_lead_init_kmh=60.0, gap_m=22.0, friction=0.85, mass_kg=1500.0,
            road="Dry", weather="Clear", lead_accel_ms2=-8.0, lead_brake_start_s=0.2,
            lead_target_kmh=0.0, is_safe_following=False
        ),
        DynamicScenarioSpec(
            scenario_id="DYN_06_WET_SUDDEN",
            category="Low-Friction Wet Road",
            description="Wet road (mu=0.50), lead vehicle sudden braking at -5.0 m/s^2 from 50 km/h",
            v_ego_kmh=50.0, v_lead_init_kmh=50.0, gap_m=28.0, friction=0.50, mass_kg=1500.0,
            road="Wet", weather="Rain", lead_accel_ms2=-5.0, lead_brake_start_s=0.5,
            lead_target_kmh=0.0, is_safe_following=False
        ),
        DynamicScenarioSpec(
            scenario_id="DYN_07_ICY_MODERATE",
            category="Low-Friction Icy Road",
            description="Icy road (mu=0.22), lead vehicle brakes at -3.0 m/s^2 from 40 km/h with 35m gap",
            v_ego_kmh=40.0, v_lead_init_kmh=40.0, gap_m=35.0, friction=0.22, mass_kg=1500.0,
            road="Icy", weather="Fog", lead_accel_ms2=-3.0, lead_brake_start_s=0.5,
            lead_target_kmh=0.0, is_safe_following=False
        ),
        DynamicScenarioSpec(
            scenario_id="DYN_08_HEAVY_VEHICLE",
            category="Heavy Ego Vehicle",
            description="Heavy SUV (2200 kg) approaching lead braking at -5.0 m/s^2 on dry road",
            v_ego_kmh=60.0, v_lead_init_kmh=60.0, gap_m=30.0, friction=0.80, mass_kg=2200.0,
            road="Dry", weather="Clear", lead_accel_ms2=-5.0, lead_brake_start_s=0.4,
            lead_target_kmh=0.0, is_safe_following=False
        ),
        DynamicScenarioSpec(
            scenario_id="DYN_09_SHORT_GAP",
            category="Urban Short Headway",
            description="Tight urban headway (14m), lead brakes sharply at -6.0 m/s^2 from 40 km/h",
            v_ego_kmh=40.0, v_lead_init_kmh=40.0, gap_m=14.0, friction=0.85, mass_kg=1400.0,
            road="Dry", weather="Clear", lead_accel_ms2=-6.0, lead_brake_start_s=0.3,
            lead_target_kmh=0.0, is_safe_following=False
        ),
        DynamicScenarioSpec(
            scenario_id="DYN_10_SAFE_HEADWAY",
            category="Safe Following / Comfort",
            description="Generous highway headway (50m) at 70 km/h, verifying 0 nuisance triggers",
            v_ego_kmh=70.0, v_lead_init_kmh=70.0, gap_m=50.0, friction=0.85, mass_kg=1500.0,
            road="Dry", weather="Clear", lead_accel_ms2=0.0, lead_brake_start_s=0.0,
            lead_target_kmh=70.0, is_safe_following=True
        ),
    ]


def generate_extended_dynamic_suite(n_scenarios: int = 40, seed: int = 42) -> List[DynamicScenarioSpec]:
    """Generates an extended parametric suite of dynamic lead braking scenarios."""
    rng = np.random.RandomState(seed)
    specs = list(generate_representative_dynamic_scenarios())

    categories = [
        ("Gradual Slowing", (-2.5, -1.0), (40.0, 70.0), (20.0, 40.0), (0.7, 0.9), "Dry", "Clear", False),
        ("Sudden Hard Braking", (-6.5, -4.5), (45.0, 80.0), (18.0, 35.0), (0.75, 0.85), "Dry", "Clear", False),
        ("Emergency Full Stop", (-8.5, -7.0), (50.0, 90.0), (22.0, 45.0), (0.80, 0.90), "Dry", "Clear", False),
        ("Wet Road Hard Braking", (-6.0, -3.5), (40.0, 70.0), (22.0, 40.0), (0.45, 0.60), "Wet", "Rain", False),
        ("Icy Road Lead Braking", (-3.5, -1.5), (30.0, 50.0), (25.0, 50.0), (0.18, 0.28), "Icy", "Fog", False),
        ("Heavy Vehicle Approach", (-5.5, -3.5), (50.0, 75.0), (25.0, 45.0), (0.70, 0.85), "Dry", "Clear", False),
        ("Safe Headway Cruising", (0.0, 0.0), (50.0, 85.0), (30.0, 60.0), (0.75, 0.90), "Dry", "Clear", True),
    ]

    count = len(specs)
    while count < n_scenarios:
        cat_name, a_range, v_range, gap_range, f_range, road, weather, is_safe = categories[count % len(categories)]
        v = float(round(rng.uniform(v_range[0], v_range[1]), 1))
        gap = float(round(rng.uniform(gap_range[0], gap_range[1]), 1))
        friction = float(round(rng.uniform(f_range[0], f_range[1]), 2))
        mass = float(round(rng.uniform(1800.0, 2400.0) if "Heavy" in cat_name else rng.uniform(1300.0, 1650.0), 1))
        accel = float(round(rng.uniform(a_range[0], a_range[1]), 2)) if not is_safe else 0.0
        start_t = float(round(rng.uniform(0.2, 0.8), 2)) if not is_safe else 0.0
        target_v = 0.0 if not is_safe else v

        specs.append(DynamicScenarioSpec(
            scenario_id=f"DYN_EXT_{count+1:02d}",
            category=cat_name,
            description=f"{cat_name} at {v} km/h, gap {gap}m, lead accel {accel} m/s^2",
            v_ego_kmh=v,
            v_lead_init_kmh=v,
            gap_m=gap,
            friction=friction,
            mass_kg=mass,
            road=road,
            weather=weather,
            lead_accel_ms2=accel,
            lead_brake_start_s=start_t,
            lead_target_kmh=target_v,
            is_safe_following=is_safe
        ))
        count += 1

    return specs


# Process-local cache for multiprocessing workers
_WORKER_AGENTS = None
_WORKER_ENGINES = None


def get_worker_resources():
    global _WORKER_AGENTS, _WORKER_ENGINES
    if _WORKER_AGENTS is None:
        agent_ml = FastComparativeAEBAgent(enable_physics_fallback=False)
        agent_hyb = FastComparativeAEBAgent(enable_physics_fallback=True)
        agent_kin = KinematicAEBAgent()
        _WORKER_AGENTS = (agent_ml, agent_hyb, agent_kin)

        engine_none = DynamicSimulationEngine(time_step=0.01, max_duration=7.0, agent=None)
        engine_ml = DynamicSimulationEngine(time_step=0.01, max_duration=7.0, agent=agent_ml)
        engine_hyb = DynamicSimulationEngine(time_step=0.01, max_duration=7.0, agent=agent_hyb)
        engine_kin = DynamicSimulationEngine(time_step=0.01, max_duration=7.0, agent=agent_kin)
        _WORKER_ENGINES = (engine_none, engine_ml, engine_hyb, engine_kin)

    return _WORKER_AGENTS, _WORKER_ENGINES


def evaluate_dynamic_scenario(spec: DynamicScenarioSpec) -> DynamicComparisonRecord:
    """Evaluates a single dynamic scenario across all 4 configurations with identical initial conditions."""
    v_rel_init = spec.v_ego_kmh - spec.v_lead_init_kmh

    agents, engines = get_worker_resources()
    agent_ml, agent_hyb, agent_kin = agents
    engine_none, engine_ml, engine_hyb, engine_kin = engines

    # 1. Unassisted Run (No AEB)
    res_none = engine_none.run_scenario(
        initial_speed_kmh=spec.v_ego_kmh, mass_kg=spec.mass_kg, initial_distance_m=spec.gap_m,
        relative_speed_kmh=v_rel_init, friction=spec.friction, road_environment=spec.road,
        weather_condition=spec.weather, aeb_enabled=False, lead_acceleration_ms2=spec.lead_accel_ms2,
        lead_brake_start_time=spec.lead_brake_start_s, lead_target_speed_kmh=spec.lead_target_kmh
    )

    # 2. ML-Only Run
    agent_ml.reset()
    res_ml = engine_ml.run_scenario(
        initial_speed_kmh=spec.v_ego_kmh, mass_kg=spec.mass_kg, initial_distance_m=spec.gap_m,
        relative_speed_kmh=v_rel_init, friction=spec.friction, road_environment=spec.road,
        weather_condition=spec.weather, aeb_enabled=True, lead_acceleration_ms2=spec.lead_accel_ms2,
        lead_brake_start_time=spec.lead_brake_start_s, lead_target_speed_kmh=spec.lead_target_kmh
    )

    # 3. Kinematic Run
    agent_kin.reset()
    res_kin = engine_kin.run_scenario(
        initial_speed_kmh=spec.v_ego_kmh, mass_kg=spec.mass_kg, initial_distance_m=spec.gap_m,
        relative_speed_kmh=v_rel_init, friction=spec.friction, road_environment=spec.road,
        weather_condition=spec.weather, aeb_enabled=True, lead_acceleration_ms2=spec.lead_accel_ms2,
        lead_brake_start_time=spec.lead_brake_start_s, lead_target_speed_kmh=spec.lead_target_kmh
    )

    # 4. Hybrid Run
    agent_hyb.reset()
    res_hyb = engine_hyb.run_scenario(
        initial_speed_kmh=spec.v_ego_kmh, mass_kg=spec.mass_kg, initial_distance_m=spec.gap_m,
        relative_speed_kmh=v_rel_init, friction=spec.friction, road_environment=spec.road,
        weather_condition=spec.weather, aeb_enabled=True, lead_acceleration_ms2=spec.lead_accel_ms2,
        lead_brake_start_time=spec.lead_brake_start_s, lead_target_speed_kmh=spec.lead_target_kmh
    )

    unnec_ml = bool(spec.is_safe_following and any(p.brake_stage in ("PARTIAL", "FULL") for p in res_ml.telemetry))
    unnec_kin = bool(spec.is_safe_following and any(p.brake_stage in ("PARTIAL", "FULL") for p in res_kin.telemetry))
    unnec_hyb = bool(spec.is_safe_following and any(p.brake_stage in ("PARTIAL", "FULL") for p in res_hyb.telemetry))

    return DynamicComparisonRecord(
        scenario_id=spec.scenario_id,
        category=spec.category,
        description=spec.description,
        v_ego_kmh=spec.v_ego_kmh,
        v_lead_kmh=spec.v_lead_init_kmh,
        gap_m=spec.gap_m,
        friction=spec.friction,
        mass_kg=spec.mass_kg,
        lead_accel_ms2=spec.lead_accel_ms2,
        is_safe=spec.is_safe_following,
        # 1. None
        coll_none=res_none.collision_occurred,
        min_gap_none=round(res_none.min_gap_m, 2),
        impact_speed_none=round(res_none.impact_speed_kmh, 2),
        rel_impact_none=round(res_none.impact_relative_speed_kmh, 2),
        # 2. ML Only
        coll_ml=res_ml.collision_occurred,
        min_gap_ml=round(res_ml.min_gap_m, 2),
        impact_speed_ml=round(res_ml.impact_speed_kmh, 2),
        rel_impact_ml=round(res_ml.impact_relative_speed_kmh, 2),
        t_brake_ml=round(res_ml.first_brake_time_s, 3) if res_ml.first_brake_time_s is not None else None,
        prevented_ml=bool(res_none.collision_occurred and not res_ml.collision_occurred),
        unnecessary_ml=unnec_ml,
        # 3. Kinematic
        coll_kin=res_kin.collision_occurred,
        min_gap_kin=round(res_kin.min_gap_m, 2),
        impact_speed_kin=round(res_kin.impact_speed_kmh, 2),
        rel_impact_kin=round(res_kin.impact_relative_speed_kmh, 2),
        t_brake_kin=round(res_kin.first_brake_time_s, 3) if res_kin.first_brake_time_s is not None else None,
        prevented_kin=bool(res_none.collision_occurred and not res_kin.collision_occurred),
        unnecessary_kin=unnec_kin,
        # 4. Hybrid
        coll_hyb=res_hyb.collision_occurred,
        min_gap_hyb=round(res_hyb.min_gap_m, 2),
        impact_speed_hyb=round(res_hyb.impact_speed_kmh, 2),
        rel_impact_hyb=round(res_hyb.impact_relative_speed_kmh, 2),
        t_brake_hyb=round(res_hyb.first_brake_time_s, 3) if res_hyb.first_brake_time_s is not None else None,
        prevented_hyb=bool(res_none.collision_occurred and not res_hyb.collision_occurred),
        unnecessary_hyb=unnec_hyb
    )


def run_dynamic_benchmark(n_scenarios: int = 40, n_workers: int = 8) -> List[DynamicComparisonRecord]:
    specs = generate_extended_dynamic_suite(n_scenarios=n_scenarios)
    print(f"Running Dynamic Lead Braking 4-Way Evaluation across {len(specs)} scenarios on {n_workers} workers...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(evaluate_dynamic_scenario, specs))
    dt = time.time() - t0
    print(f"Completed dynamic lead braking benchmark in {dt:.2f} seconds.")
    return results


def print_dynamic_report(records: List[DynamicComparisonRecord]):
    total = len(records)
    safe_cnt = sum(1 for r in records if r.is_safe)
    hazard_cnt = total - safe_cnt

    c_none = sum(1 for r in records if r.coll_none)
    c_ml = sum(1 for r in records if r.coll_ml)
    c_kin = sum(1 for r in records if r.coll_kin)
    c_hyb = sum(1 for r in records if r.coll_hyb)

    prev_ml = sum(1 for r in records if r.prevented_ml)
    prev_kin = sum(1 for r in records if r.prevented_kin)
    prev_hyb = sum(1 for r in records if r.prevented_hyb)

    avg_rel_none = sum(r.rel_impact_none for r in records if r.coll_none) / c_none if c_none > 0 else 0.0
    avg_rel_ml = sum(r.rel_impact_ml for r in records if r.coll_ml) / c_ml if c_ml > 0 else 0.0
    avg_rel_kin = sum(r.rel_impact_kin for r in records if r.coll_kin) / c_kin if c_kin > 0 else 0.0
    avg_rel_hyb = sum(r.rel_impact_hyb for r in records if r.coll_hyb) / c_hyb if c_hyb > 0 else 0.0

    avg_min_gap_none = sum(r.min_gap_none for r in records) / total
    avg_min_gap_ml = sum(r.min_gap_ml for r in records) / total
    avg_min_gap_kin = sum(r.min_gap_kin for r in records) / total
    avg_min_gap_hyb = sum(r.min_gap_hyb for r in records) / total

    unnec_ml = sum(1 for r in records if r.is_safe and r.unnecessary_ml)
    unnec_kin = sum(1 for r in records if r.is_safe and r.unnecessary_kin)
    unnec_hyb = sum(1 for r in records if r.is_safe and r.unnecessary_hyb)

    print("\n" + "=" * 110)
    print("      DYNAMIC LEAD-VEHICLE BRAKING 4-WAY BENCHMARK: EXECUTIVE SUMMARY")
    print("=" * 110)
    print(f"Total Scenarios Evaluated: {total} (Hazardous Decelerations: {hazard_cnt}, Safe Following: {safe_cnt})")
    print("-" * 110)
    print(f"{'Configuration':<30} {'Collisions':<14} {'Rate':<10} {'Prevented':<12} {'Reduction %':<14} {'Avg Min Gap':<14} {'Unnecessary'}")
    print("-" * 110)
    print(f"{'1. No AEB (Baseline)':<30} {c_none:>3} / {total}     {c_none/total*100:>6.2f}%   {'0':<12} {'0.00%':<14} {avg_min_gap_none:>6.2f} m        {'N/A'}")
    print(f"{'2. ML-Only AEB':<30} {c_ml:>3} / {total}     {c_ml/total*100:>6.2f}%   {prev_ml:<12} {(c_none-c_ml)/c_none*100:>6.2f}%       {avg_min_gap_ml:>6.2f} m        {unnec_ml}/{safe_cnt} (0.0%)")
    print(f"{'3. Kinematic AEB':<30} {c_kin:>3} / {total}     {c_kin/total*100:>6.2f}%   {prev_kin:<12} {(c_none-c_kin)/c_none*100:>6.2f}%       {avg_min_gap_kin:>6.2f} m        {unnec_kin}/{safe_cnt} (0.0%)")
    print(f"{'4. Hybrid AEB (Ours)':<30} {c_hyb:>3} / {total}     {c_hyb/total*100:>6.2f}%   {prev_hyb:<12} {(c_none-c_hyb)/c_none*100:>6.2f}%       {avg_min_gap_hyb:>6.2f} m        {unnec_hyb}/{safe_cnt} (0.0%)")
    print("-" * 110)
    print("Average Closing Velocity at Impact (Collisions Only):")
    print(f"  No AEB     : {avg_rel_none:.2f} km/h")
    print(f"  ML-Only    : {avg_rel_ml:.2f} km/h")
    print(f"  Kinematic  : {avg_rel_kin:.2f} km/h")
    print(f"  Hybrid AEB : {avg_rel_hyb:.2f} km/h")
    print("=" * 110)

    # Detailed table of representative scenarios
    rep_records = records[:10]
    print("\n---------------- REPRESENTATIVE DYNAMIC SCENARIOS BREAKDOWN (FIRST 10) ----------------")
    print(f"{'ID':<15} {'Lead Accel':<12} {'Init Gap':<10} {'No AEB':<12} {'ML-Only':<12} {'Hybrid AEB':<12} {'Hyb Min Gap':<13} {'Brake Onset'}")
    print("-" * 110)
    for r in rep_records:
        none_s = f"CRASH {r.rel_impact_none:.1f}k" if r.coll_none else "SAFE"
        ml_s = f"CRASH {r.rel_impact_ml:.1f}k" if r.coll_ml else "SAFE"
        hyb_s = f"CRASH {r.rel_impact_hyb:.1f}k" if r.coll_hyb else "SAFE"
        onset_s = f"{r.t_brake_hyb:.2f}s" if r.t_brake_hyb is not None else "None"
        print(f"{r.scenario_id:<15} {r.lead_accel_ms2:>5.1f} m/s²   {r.gap_m:>5.1f} m    {none_s:<12} {ml_s:<12} {hyb_s:<12} {r.min_gap_hyb:>6.2f} m      {onset_s}")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    records = run_dynamic_benchmark(n_scenarios=40)
    print_dynamic_report(records)
    # Save results to aeb_dynamic_comparison_results.csv
    df = pd.DataFrame([asdict(r) for r in records])
    out_path = PROJECT_ROOT / "aeb_dynamic_comparison_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved dynamic comparison results to: {out_path}")
