import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiment_close_obstacle_baseline import KinematicAEBAgent
from braking_controller import BrakeStage
from simulation_engine import SimulationEngine, SimulationOutcome


def test_kinematic_agent_initialization():
    """Verify initialization defaults of the experimental kinematic agent."""
    agent = KinematicAEBAgent()
    assert agent.brake_force_max == 12000.0
    assert agent.latency == 0.2
    assert agent.margin_critical == 1.0
    assert agent.margin_warning == 4.0
    assert agent.latched_stage == BrakeStage.OFF


def test_kinematic_agent_close_obstacle_full_braking():
    """Verify that close hazardous obstacles trigger BrakeStage.FULL deterministically."""
    agent = KinematicAEBAgent()
    decision = agent.evaluate_state(
        vehicle_speed_kmh=50.0,
        mass_kg=1500.0,
        distance_m=4.0,
        relative_speed_kmh=20.0,
        friction=0.8
    )
    assert decision.stage == BrakeStage.FULL
    assert "Kinematic FULL" in decision.reason
    assert decision.raw_prediction == 1


def test_kinematic_agent_safe_following_no_intervention():
    """Verify that equal-speed following or opening gaps never trigger braking."""
    agent = KinematicAEBAgent()
    # Speed matched (v_rel = 0)
    decision = agent.evaluate_state(
        vehicle_speed_kmh=60.0,
        mass_kg=1500.0,
        distance_m=15.0,
        relative_speed_kmh=0.0,
        friction=0.8
    )
    assert decision.stage == BrakeStage.OFF
    assert "moving away or speed matched" in decision.reason

    # Obstacle receding (negative v_rel)
    dec_receding = agent.evaluate_state(
        vehicle_speed_kmh=60.0,
        mass_kg=1500.0,
        distance_m=10.0,
        relative_speed_kmh=-15.0,
        friction=0.8
    )
    assert dec_receding.stage == BrakeStage.OFF


def test_kinematic_agent_no_division_by_zero():
    """Verify that zero vehicle speed, zero distance, or zero relative speed do not cause ZeroDivisionError."""
    agent = KinematicAEBAgent()
    # Stopped vehicle
    dec1 = agent.evaluate_state(0.0, 1500.0, 10.0, 0.0, 0.8)
    assert dec1.stage == BrakeStage.OFF

    # Collision already occurred (distance 0)
    dec2 = agent.evaluate_state(40.0, 1500.0, 0.0, 40.0, 0.8)
    assert dec2.stage == BrakeStage.OFF

    # Zero relative speed with close distance
    dec3 = agent.evaluate_state(50.0, 1500.0, 5.0, 0.0, 0.8)
    assert dec3.stage == BrakeStage.OFF


def test_kinematic_agent_simulation_integration():
    """Verify that KinematicAEBAgent integrates cleanly into SimulationEngine closed-loop runs."""
    agent = KinematicAEBAgent()
    engine = SimulationEngine(time_step=0.01, agent=agent)
    result = engine.run_scenario(
        initial_speed_kmh=40.0,
        mass_kg=1500.0,
        initial_distance_m=20.0,
        relative_speed_kmh=40.0,
        friction=0.85,
        aeb_enabled=True
    )
    assert result.outcome == SimulationOutcome.SAFE_STOP
    assert result.collision_occurred is False
    assert result.final_gap_m > 0.0

