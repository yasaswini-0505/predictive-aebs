import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aeb_agent import AEBAgent
from braking_controller import BrakeStage
from simulation_engine import SimulationEngine, SimulationOutcome


def test_close_obstacle_low_relative_speed_triggers_fallback():
    """Verify that a close obstacle with low ML collision probability still triggers physics safety fallback."""
    agent = AEBAgent(enable_physics_fallback=True)
    # 4m distance, 20 km/h closing speed: ML predicts very low risk, but gap <= d_crit
    decision = agent.evaluate_state(
        vehicle_speed_kmh=50.0,
        mass_kg=1500.0,
        distance_m=4.0,
        relative_speed_kmh=20.0,
        friction=0.8,
        road_environment="Dry",
        weather_condition="Clear"
    )
    assert decision.stage == BrakeStage.FULL
    assert "Physics safety fallback" in decision.reason


def test_equal_speeds_and_opening_gaps_no_braking():
    """Verify that matching speeds or receding vehicles never trigger partial or full braking."""
    agent = AEBAgent()
    # Speed matched (v_rel = 0)
    dec1 = agent.evaluate_state(
        vehicle_speed_kmh=60.0, mass_kg=1500.0, distance_m=10.0,
        relative_speed_kmh=0.0, friction=0.85
    )
    assert dec1.stage == BrakeStage.OFF
    assert "moving away or speed matched" in dec1.reason

    # Opening gap (v_rel < 0)
    dec2 = agent.evaluate_state(
        vehicle_speed_kmh=60.0, mass_kg=1500.0, distance_m=8.0,
        relative_speed_kmh=-10.0, friction=0.85
    )
    assert dec2.stage == BrakeStage.OFF


def test_heavy_vehicle_stopping_distance_adaptation():
    """Verify that vehicle mass increases critical stopping distance envelope."""
    agent = AEBAgent()
    d_crit_light, _ = agent.calculate_stopping_distances(mass_kg=1200.0, friction=0.8, relative_speed_kmh=50.0)
    d_crit_heavy, _ = agent.calculate_stopping_distances(mass_kg=2400.0, friction=0.8, relative_speed_kmh=50.0)
    assert d_crit_heavy > d_crit_light, "Heavier vehicle must require a larger critical stopping distance"


def test_high_speed_closing_scenario():
    """Verify that high-speed highway approach triggers emergency braking."""
    agent = AEBAgent()
    decision = agent.evaluate_state(
        vehicle_speed_kmh=90.0,
        mass_kg=1500.0,
        distance_m=35.0,
        relative_speed_kmh=90.0,
        friction=0.85,
        road_environment="Dry",
        weather_condition="Clear"
    )
    assert decision.stage == BrakeStage.FULL


def test_low_friction_road_stopping_distance_adaptation():
    """Verify that low road friction (icy/wet) significantly expands stopping distances."""
    agent = AEBAgent()
    d_crit_dry, _ = agent.calculate_stopping_distances(mass_kg=1500.0, friction=0.85, relative_speed_kmh=50.0)
    d_crit_icy, _ = agent.calculate_stopping_distances(mass_kg=1500.0, friction=0.20, relative_speed_kmh=50.0)
    assert d_crit_icy > 2.0 * d_crit_dry, "Icy road stopping distance must be more than double dry road"


def test_moving_obstacle_speed_matching():
    """Verify progressive multi-stage braking when approaching a slower moving lead vehicle."""
    agent = AEBAgent()
    # 1. Approaching vehicle with gap outside warning horizon (18m, closing at 25 km/h, TTC=2.59s > 2.4s)
    dec_far = agent.evaluate_state(
        vehicle_speed_kmh=70.0,
        mass_kg=1500.0,
        distance_m=18.0,
        relative_speed_kmh=25.0,
        friction=0.80
    )
    assert dec_far.stage == BrakeStage.OFF, "Outside warning horizon should remain OFF"

    # 2. As gap closes into warning horizon (12m, closing at 25 km/h, TTC=1.73s <= 2.4s)
    dec_warn = agent.evaluate_state(
        vehicle_speed_kmh=70.0,
        mass_kg=1500.0,
        distance_m=12.0,
        relative_speed_kmh=25.0,
        friction=0.80
    )
    assert dec_warn.stage == BrakeStage.PARTIAL, "Within warning horizon must trigger PARTIAL"

    # 3. Inside critical stopping distance (8m, TTC=1.15s <= 1.4s)
    dec_crit = agent.evaluate_state(
        vehicle_speed_kmh=70.0,
        mass_kg=1500.0,
        distance_m=8.0,
        relative_speed_kmh=25.0,
        friction=0.80
    )
    assert dec_crit.stage == BrakeStage.FULL, "Within critical distance must escalate to FULL"


def test_pure_ml_behavior_when_fallback_disabled():
    """Verify that disabling the physics fallback leaves decisions solely to the ML model."""
    agent_ml_only = AEBAgent(enable_physics_fallback=False)
    assert agent_ml_only.enable_physics_fallback is False

    # Close obstacle with low ML probability should NOT trigger physics fallback when disabled
    decision = agent_ml_only.evaluate_state(
        vehicle_speed_kmh=30.0,
        mass_kg=1500.0,
        distance_m=4.0,
        relative_speed_kmh=10.0,
        friction=0.8,
        road_environment="Dry",
        weather_condition="Clear"
    )
    assert "Physics safety fallback" not in decision.reason


def test_safe_following_no_unnecessary_braking():
    """Verify that a full closed-loop run of car-following produces zero nuisance brake activations."""
    agent = AEBAgent()
    engine = SimulationEngine(time_step=0.01, max_duration=3.0, agent=agent)
    result = engine.run_scenario(
        initial_speed_kmh=55.0,
        mass_kg=1500.0,
        initial_distance_m=25.0,
        relative_speed_kmh=0.0,
        friction=0.85,
        aeb_enabled=True
    )
    assert result.collision_occurred is False
    assert all(p.brake_stage == "OFF" for p in result.telemetry), "Must produce zero unnecessary braking"


def test_stage_escalation_and_hysteresis():
    """Verify stage escalation from warning to emergency and persistent hysteresis holding."""
    agent = AEBAgent()
    # Trigger full emergency
    d1 = agent.evaluate_state(
        vehicle_speed_kmh=50.0, mass_kg=1500.0, distance_m=8.0,
        relative_speed_kmh=35.0, friction=0.80
    )
    assert d1.stage == BrakeStage.FULL
    assert agent.latched_stage == BrakeStage.FULL

    # Vehicle slows down, gap still closing => hysteresis must hold FULL
    d2 = agent.evaluate_state(
        vehicle_speed_kmh=25.0, mass_kg=1500.0, distance_m=6.0,
        relative_speed_kmh=15.0, friction=0.80
    )
    assert d2.stage == BrakeStage.FULL
    assert "Held by AEB hysteresis" in d2.reason


def test_stationary_obstacle_complete_stop():
    """Verify that a standard 50 km/h scenario on dry road safely halts before the obstacle."""
    agent = AEBAgent()
    engine = SimulationEngine(time_step=0.01, agent=agent)
    result = engine.run_scenario(
        initial_speed_kmh=50.0,
        mass_kg=1500.0,
        initial_distance_m=30.0,
        relative_speed_kmh=50.0,
        friction=0.80,
        aeb_enabled=True
    )
    assert result.outcome == SimulationOutcome.SAFE_STOP
    assert result.collision_occurred is False
    assert result.final_speed_kmh == 0.0
    assert result.final_gap_m > 0.0
