import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulation_engine import SimulationEngine, SimulationOutcome, SimulationResult
from aeb_agent import AEBAgent


def test_simulation_input_validation():
    """Verify ValueError is raised for invalid initial kinematic parameters."""
    engine = SimulationEngine()
    with pytest.raises(ValueError, match="Initial vehicle speed cannot be negative"):
        engine.run_scenario(initial_speed_kmh=-10.0, mass_kg=1500.0, initial_distance_m=20.0, relative_speed_kmh=10.0, friction=0.8)

    with pytest.raises(ValueError, match="Initial distance must be positive"):
        engine.run_scenario(initial_speed_kmh=50.0, mass_kg=1500.0, initial_distance_m=0.0, relative_speed_kmh=10.0, friction=0.8)

    with pytest.raises(ValueError, match="Vehicle mass must be positive"):
        engine.run_scenario(initial_speed_kmh=50.0, mass_kg=-500.0, initial_distance_m=20.0, relative_speed_kmh=10.0, friction=0.8)

    with pytest.raises(ValueError, match="Friction coefficient must be positive"):
        engine.run_scenario(initial_speed_kmh=50.0, mass_kg=1500.0, initial_distance_m=20.0, relative_speed_kmh=10.0, friction=-0.2)


def test_simulation_disabled_aeb_collision():
    """Verify that unassisted vehicle (driver inattention) collides with stationary obstacle."""
    engine = SimulationEngine()
    result = engine.run_scenario(
        initial_speed_kmh=50.0,
        mass_kg=1500.0,
        initial_distance_m=15.0,
        relative_speed_kmh=50.0,
        friction=0.85,
        aeb_enabled=False
    )
    assert result.collision_occurred is True
    assert result.outcome == SimulationOutcome.COLLISION
    assert result.impact_speed_kmh > 40.0
    assert result.final_gap_m == 0.0


def test_simulation_enabled_aeb_prevents_collision():
    """Verify that AEB intervention prevents collision and safely stops on dry road with adequate initial gap."""
    agent = AEBAgent()
    engine = SimulationEngine(agent=agent)
    result = engine.run_scenario(
        initial_speed_kmh=45.0,
        mass_kg=1400.0,
        initial_distance_m=25.0,
        relative_speed_kmh=45.0,
        friction=0.85,
        aeb_enabled=True
    )
    assert result.collision_occurred is False
    assert result.outcome == SimulationOutcome.SAFE_STOP
    assert result.final_speed_kmh == 0.0
    assert result.final_gap_m > 0.0


def test_simulation_enabled_aeb_mitigates_impact():
    """Verify that in physically unavoidable scenarios, AEB significantly reduces impact velocity."""
    agent = AEBAgent()
    engine = SimulationEngine(agent=agent)

    # Extreme scenario: 90 km/h, gap only 12m, icy road (mu=0.20)
    res_none = engine.run_scenario(
        initial_speed_kmh=90.0, mass_kg=1500.0, initial_distance_m=12.0,
        relative_speed_kmh=90.0, friction=0.20, aeb_enabled=False
    )
    res_aeb = engine.run_scenario(
        initial_speed_kmh=90.0, mass_kg=1500.0, initial_distance_m=12.0,
        relative_speed_kmh=90.0, friction=0.20, aeb_enabled=True
    )

    assert res_none.collision_occurred is True
    assert res_aeb.collision_occurred is True
    # Impact velocity under AEB must be less than or equal to unassisted baseline
    assert res_aeb.impact_speed_kmh <= res_none.impact_speed_kmh


def test_simulation_moving_lead_vehicle_safe_following():
    """Verify that lead vehicle moving at equal speed is safely followed without false collision."""
    agent = AEBAgent()
    engine = SimulationEngine(agent=agent)
    # Both traveling at 60 km/h (relative speed = 0) with 20m gap
    result = engine.run_scenario(
        initial_speed_kmh=60.0,
        mass_kg=1500.0,
        initial_distance_m=20.0,
        relative_speed_kmh=0.0,
        friction=0.85,
        aeb_enabled=True
    )
    assert result.collision_occurred is False
    assert not any(p.brake_stage in ["PARTIAL", "FULL"] for p in result.telemetry)


def test_simulation_negative_speed_prevention():
    """Verify that vehicle speed never dips below 0 m/s (no backward motion under deceleration)."""
    agent = AEBAgent()
    engine = SimulationEngine(agent=agent)
    result = engine.run_scenario(
        initial_speed_kmh=30.0,
        mass_kg=1500.0,
        initial_distance_m=30.0,
        relative_speed_kmh=30.0,
        friction=0.85,
        aeb_enabled=True
    )
    for p in result.telemetry:
        assert p.vehicle_speed_kmh >= 0.0, "Ego speed must never be negative"


def test_simulation_relative_speed_sign_convention():
    """Verify that positive relative speed closes the gap and lead vehicle speed is properly calculated."""
    engine = SimulationEngine()
    # v_ego = 60 km/h, v_rel = 20 km/h => v_obs = 40 km/h
    res = engine.run_scenario(
        initial_speed_kmh=60.0, mass_kg=1500.0, initial_distance_m=30.0,
        relative_speed_kmh=20.0, friction=0.85, aeb_enabled=False
    )
    assert res.telemetry[0].obstacle_speed_kmh == pytest.approx(40.0, abs=0.1)
    # Gap should decrease over initial steps
    assert res.telemetry[10].gap_m < res.telemetry[0].gap_m

