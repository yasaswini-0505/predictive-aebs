import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aeb_agent import AEBAgent, AEBDecision
from braking_controller import BrakeStage


def test_agent_initialization():
    """Verify default thresholds, feature list, and model loading in AEBAgent."""
    agent = AEBAgent()
    assert agent.partial_risk_threshold == 0.35
    assert agent.full_risk_threshold == 0.60
    assert agent.ttc_critical == 1.4
    assert agent.ttc_warning == 2.4
    assert agent.hysteresis_hold is True
    assert agent.enable_physics_fallback is True
    assert agent.latched_stage == BrakeStage.OFF
    assert len(agent.FEATURE_COLUMNS) == 7


def test_agent_emergency_scenario_evaluation():
    """Verify that an imminent collision hazard triggers BrakeStage.FULL."""
    agent = AEBAgent()
    # High ego speed, small gap, high relative speed
    decision = agent.evaluate_state(
        vehicle_speed_kmh=80.0,
        mass_kg=1500.0,
        distance_m=12.0,
        relative_speed_kmh=60.0,
        friction=0.8,
        road_environment="Dry",
        weather_condition="Clear"
    )
    assert isinstance(decision, AEBDecision)
    assert decision.stage == BrakeStage.FULL
    assert decision.collision_probability > 0.50 or decision.ttc <= 1.4
    assert decision.ttc < 1.0


def test_agent_safe_scenario_evaluation():
    """Verify that distant obstacles or matched speeds evaluate to BrakeStage.OFF."""
    agent = AEBAgent()
    # Low speed, very large distance
    decision = agent.evaluate_state(
        vehicle_speed_kmh=30.0,
        mass_kg=1400.0,
        distance_m=80.0,
        relative_speed_kmh=10.0,
        friction=0.85,
        road_environment="Dry",
        weather_condition="Clear"
    )
    assert decision.stage == BrakeStage.OFF
    assert decision.collision_probability < 0.35
    assert decision.ttc > 5.0


def test_agent_ttc_calculation():
    """Verify Time-To-Collision calculation for closing, stationary, and receding conditions."""
    agent = AEBAgent()
    # Closing obstacle: 20m gap, closing at 72 km/h (20 m/s) => TTC = 1.0 s
    ttc = agent.compute_ttc(distance_m=20.0, relative_speed_kmh=72.0)
    assert pytest.approx(ttc, abs=1e-3) == 1.0

    # Non-closing obstacle: relative speed zero (speed matched) => TTC = infinity
    ttc_matched = agent.compute_ttc(distance_m=20.0, relative_speed_kmh=0.0)
    assert ttc_matched == float("inf")

    # Obstacle moving away: negative relative speed => TTC = infinity
    ttc_receding = agent.compute_ttc(distance_m=20.0, relative_speed_kmh=-20.0)
    assert ttc_receding == float("inf")


def test_agent_hysteresis_latching():
    """Verify that once FULL braking is latched, it is held until the vehicle comes to a stop."""
    agent = AEBAgent()
    # Trigger FULL braking
    dec1 = agent.evaluate_state(
        vehicle_speed_kmh=70.0,
        mass_kg=1500.0,
        distance_m=10.0,
        relative_speed_kmh=50.0,
        friction=0.8
    )
    assert dec1.stage == BrakeStage.FULL
    assert agent.latched_stage == BrakeStage.FULL

    # Vehicle slows down, but is still moving and closing gap
    dec2 = agent.evaluate_state(
        vehicle_speed_kmh=30.0,
        mass_kg=1500.0,
        distance_m=8.0,
        relative_speed_kmh=20.0,
        friction=0.8
    )
    assert dec2.stage == BrakeStage.FULL
    assert "Held by AEB hysteresis" in dec2.reason

    # Vehicle halts (speed <= 0.1 km/h) => hysteresis unlatches
    dec3 = agent.evaluate_state(
        vehicle_speed_kmh=0.0,
        mass_kg=1500.0,
        distance_m=3.0,
        relative_speed_kmh=0.0,
        friction=0.8
    )
    assert dec3.stage == BrakeStage.OFF
    assert agent.latched_stage == BrakeStage.OFF

