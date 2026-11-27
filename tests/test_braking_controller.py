import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from braking_controller import BrakingController, BrakeStage


def test_invalid_parameters():
    """Verify ValueError is raised for physically invalid initial parameters."""
    with pytest.raises(ValueError, match="mass must be positive"):
        BrakingController(mass=-100.0)
    with pytest.raises(ValueError, match="mass must be positive"):
        BrakingController(mass=0.0)

    with pytest.raises(ValueError, match="Friction coefficient must be positive"):
        BrakingController(friction=-0.5)
    with pytest.raises(ValueError, match="Friction coefficient must be positive"):
        BrakingController(friction=0.0)

    with pytest.raises(ValueError, match="Max brake force must be positive"):
        BrakingController(max_brake_force=-5000.0)

    with pytest.raises(ValueError, match="Response delay cannot be negative"):
        BrakingController(response_delay=-0.1)


def test_max_available_deceleration_force_limited():
    """Verify that when tire-road friction is abundant, actuator force limits deceleration."""
    # mass = 2000 kg, F_max = 12000 N => a_force = 6.0 m/s^2
    # friction = 0.9 => a_friction = 0.9 * 9.81 = 8.829 m/s^2
    # a_max = min(6.0, 8.829) = 6.0 m/s^2
    controller = BrakingController(mass=2000.0, friction=0.9, max_brake_force=12000.0)
    assert pytest.approx(controller.max_available_deceleration, abs=1e-4) == 6.0


def test_max_available_deceleration_friction_limited():
    """Verify that under low road friction, tire adhesion limits deceleration."""
    # mass = 1200 kg, F_max = 12000 N => a_force = 10.0 m/s^2
    # friction = 0.3 => a_friction = 0.3 * 9.81 = 2.943 m/s^2
    # a_max = min(10.0, 2.943) = 2.943 m/s^2
    controller = BrakingController(mass=1200.0, friction=0.3, max_brake_force=12000.0)
    assert pytest.approx(controller.max_available_deceleration, abs=1e-4) == 2.943


def test_multi_stage_braking():
    """Verify deceleration scaling across OFF (0%), PARTIAL (40%), and FULL (100%) stages."""
    controller = BrakingController(mass=1500.0, friction=0.8, max_brake_force=12000.0, response_delay=0.2)
    # a_force = 8.0, a_friction = 0.8 * 9.81 = 7.848 => a_max = 7.848 m/s^2
    a_max = controller.max_available_deceleration

    # 1. OFF stage
    controller.set_target_stage(BrakeStage.OFF, current_time=0.0)
    assert controller.compute_deceleration(current_time=0.5, current_speed_ms=15.0) == 0.0

    # 2. PARTIAL stage after response delay
    controller.set_target_stage(BrakeStage.PARTIAL, current_time=1.0)
    a_partial = controller.compute_deceleration(current_time=1.3, current_speed_ms=15.0)
    assert pytest.approx(a_partial, abs=1e-4) == 0.4 * a_max

    # 3. FULL stage
    controller.set_target_stage(BrakeStage.FULL, current_time=2.0)
    a_full = controller.compute_deceleration(current_time=2.3, current_speed_ms=15.0)
    assert pytest.approx(a_full, abs=1e-4) == 1.0 * a_max


def test_reset():
    """Verify controller reset restores OFF stage, clears activation timestamp, and deactivates."""
    controller = BrakingController()
    controller.set_target_stage(BrakeStage.FULL, current_time=1.0)
    assert controller.is_active is True
    assert controller.target_stage == BrakeStage.FULL

    controller.reset()
    assert controller.current_stage == BrakeStage.OFF
    assert controller.target_stage == BrakeStage.OFF
    assert controller.activation_time is None
    assert controller.is_active is False


def test_response_delay_latency():
    """Verify that actuator response latency suppresses deceleration during the delay window."""
    controller = BrakingController(response_delay=0.2)
    controller.set_target_stage(BrakeStage.FULL, current_time=1.0)

    # Within latency window [1.0s, 1.20s): deceleration must be zero
    assert controller.compute_deceleration(current_time=1.05, current_speed_ms=20.0) == 0.0
    assert controller.compute_deceleration(current_time=1.19, current_speed_ms=20.0) == 0.0

    # At or after latency window (t >= 1.20s): full deceleration engages
    a_engaged = controller.compute_deceleration(current_time=1.201, current_speed_ms=20.0)
    assert a_engaged > 0.0
    assert pytest.approx(a_engaged, abs=1e-3) == controller.max_available_deceleration


def test_zero_speed_cutoff():
    """Verify that deceleration drops to 0.0 when vehicle speed reaches 0 to avoid reversing."""
    controller = BrakingController(response_delay=0.2)
    controller.set_target_stage(BrakeStage.FULL, current_time=0.0)

    # Vehicle stopped (speed = 0.0)
    assert controller.compute_deceleration(current_time=0.5, current_speed_ms=0.0) == 0.0
    # Negative speed (numerical undershoot)
    assert controller.compute_deceleration(current_time=0.5, current_speed_ms=-0.1) == 0.0

