"""
Automated Unit Tests for Experimental Predictive AEB Agent.

Verifies:
1. Predictive AEB Agent initialization and Cython tree inference.
2. Early hazard identification from lead deceleration.
3. Brake light intent perception and projected TTC computation.
4. Zero false braking in safe following and opening gap conditions.
5. Direct verification that Predictive ML triggers earlier than Physics-Only AEB.
6. Verification that Predictive Hybrid avoids collisions where Physics-Only crashes.
7. Robustness against sensor noise and friction uncertainty.
"""

import pytest
import numpy as np

from predictive_aeb_agent import PredictiveAEBAgent
from experiment_close_obstacle_baseline import KinematicAEBAgent
from dynamic_simulation_engine import DynamicSimulationEngine
from simulation_engine import SimulationOutcome
from braking_controller import BrakeStage
from predictive_aeb_dataset_generator import compute_projected_ttc


class TestPredictiveAEB:
    """Test suite for Predictive AEB implementation."""

    @pytest.fixture
    def predictive_agent(self):
        """Creates a PredictiveAEBAgent with hybrid physics fallback active."""
        return PredictiveAEBAgent(enable_physics_fallback=True)

    @pytest.fixture
    def dynamic_engine(self, predictive_agent):
        """Creates a DynamicSimulationEngine attached to the predictive agent."""
        return DynamicSimulationEngine(time_step=0.01, max_duration=6.0, agent=predictive_agent)

    def test_predictive_agent_initialization(self, predictive_agent):
        """Verifies model loading and feature dimensions."""
        assert predictive_agent.pipeline is not None
        assert predictive_agent.clf is not None
        assert len(predictive_agent.trees) == 100
        assert predictive_agent.is_predictive is True

    def test_projected_ttc_calculation(self):
        """Verifies quadratic acceleration-corrected TTC computation."""
        # Gap = 20m, v_rel = 0 m/s, a_rel = +5 m/s^2 (lead decelerating at -5 m/s^2)
        # 0.5 * 5 * t^2 = 20 => 2.5 t^2 = 20 => t^2 = 8 => t = 2.83s
        ttc_proj = compute_projected_ttc(gap_m=20.0, v_rel_ms=0.0, a_rel_ms2=5.0)
        assert abs(ttc_proj - np.sqrt(8.0)) < 0.05

        # Opening gap (a_rel = -2.0 m/s^2, v_rel = 0): should return 999.0
        ttc_open = compute_projected_ttc(gap_m=20.0, v_rel_ms=0.0, a_rel_ms2=-2.0)
        assert ttc_open >= 999.0

    def test_early_hazard_from_lead_deceleration(self, predictive_agent):
        """
        Verifies that lead vehicle braking (-6 m/s^2) triggers high risk
        at 14m headway even when relative speed is still near zero.
        """
        # Snapshot at start of braking: v_rel = 1 km/h, gap = 14m, lead braking at -6 m/s^2
        decision = predictive_agent.evaluate_state(
            vehicle_speed_kmh=60.0,
            mass_kg=1500.0,
            distance_m=14.0,
            relative_speed_kmh=1.0,
            friction=0.85,
            lead_deceleration_ms2=-6.0,
            brake_light_detected=1
        )

        assert decision.collision_probability >= 0.70
        assert decision.stage in (BrakeStage.PARTIAL, BrakeStage.FULL)
        assert "Predictive ML" in decision.reason

    def test_safe_following_no_false_braking(self, predictive_agent):
        """Verifies zero false alarms in safe cruising and opening gaps."""
        # Cruising at matched speed
        dec_cruise = predictive_agent.evaluate_state(
            vehicle_speed_kmh=70.0,
            mass_kg=1500.0,
            distance_m=35.0,
            relative_speed_kmh=0.0,
            friction=0.85,
            lead_deceleration_ms2=0.0,
            brake_light_detected=0
        )
        assert dec_cruise.stage == BrakeStage.OFF
        assert dec_cruise.collision_probability < 0.20

        # Lead accelerating away (+2 m/s^2)
        dec_accel = predictive_agent.evaluate_state(
            vehicle_speed_kmh=70.0,
            mass_kg=1500.0,
            distance_m=30.0,
            relative_speed_kmh=0.0,
            friction=0.85,
            lead_deceleration_ms2=2.0,
            brake_light_detected=0
        )
        assert dec_accel.stage == BrakeStage.OFF

    def test_predictive_triggers_earlier_than_physics(self):
        """
        Directly compares activation onset: Predictive ML must trigger
        substantially earlier than Physics-Only AEB during sudden lead braking.
        """
        # Scenario: 60 km/h, gap = 25m, lead hard brakes at -6.0 m/s^2 starting at t=0.3s
        eng_phys = DynamicSimulationEngine(time_step=0.01, max_duration=6.0, agent=KinematicAEBAgent())
        res_phys = eng_phys.run_scenario(
            60.0, 1500.0, 25.0, 0.0, 0.85, lead_acceleration_ms2=-6.0,
            lead_brake_start_time=0.3, aeb_enabled=True, record_telemetry=False
        )

        eng_pred = DynamicSimulationEngine(time_step=0.01, max_duration=6.0, agent=PredictiveAEBAgent(enable_physics_fallback=True))
        res_pred = eng_pred.run_scenario(
            60.0, 1500.0, 25.0, 0.0, 0.85, lead_acceleration_ms2=-6.0,
            lead_brake_start_time=0.3, aeb_enabled=True, record_telemetry=False
        )

        assert res_phys.first_brake_time_s is not None
        assert res_pred.first_brake_time_s is not None
        # Predictive ML must activate earlier by at least 0.35 seconds
        lead_time = res_phys.first_brake_time_s - res_pred.first_brake_time_s
        assert lead_time >= 0.35, f"Expected lead time >= 0.35s, got {lead_time:.3f}s"
        # Predictive ML must maintain a much larger minimum gap
        assert res_pred.min_gap_m > res_phys.min_gap_m + 5.0

    def test_predictive_prevents_collision_where_physics_crashes(self):
        """
        Validates boundary scenario where Physics-Only reacts too late and crashes,
        while Predictive Hybrid brakes early and safely stops without collision.
        """
        # 70 km/h, 20m gap, lead brakes hard at -6.0 m/s^2 at t=0.3s
        eng_phys = DynamicSimulationEngine(time_step=0.01, max_duration=6.0, agent=KinematicAEBAgent())
        res_phys = eng_phys.run_scenario(
            70.0, 1500.0, 20.0, 0.0, 0.85, lead_acceleration_ms2=-6.0,
            lead_brake_start_time=0.3, aeb_enabled=True, record_telemetry=False
        )

        eng_pred = DynamicSimulationEngine(time_step=0.01, max_duration=6.0, agent=PredictiveAEBAgent(enable_physics_fallback=True))
        res_pred = eng_pred.run_scenario(
            70.0, 1500.0, 20.0, 0.0, 0.85, lead_acceleration_ms2=-6.0,
            lead_brake_start_time=0.3, aeb_enabled=True, record_telemetry=False
        )

        assert res_phys.collision_occurred is True, "Physics-only should collide in this tight scenario"
        assert res_pred.collision_occurred is False, "Predictive AEB must safely prevent the collision"
        assert res_pred.outcome == SimulationOutcome.SAFE_STOP
        assert res_pred.min_gap_m > 0.5

    def test_sensor_uncertainty_robustness(self, predictive_agent):
        """
        Verifies that noisy perception inputs (friction +/- 15%, accel +/- 0.5 m/s^2)
        maintain stable, consistent braking decisions without chattering.
        """
        rng = np.random.RandomState(42)
        stages = []
        for _ in range(20):
            noisy_mu = float(0.85 * (1.0 + rng.normal(0, 0.10)))
            noisy_a = float(-6.0 + rng.normal(0, 0.30))
            dec = predictive_agent.evaluate_state(
                vehicle_speed_kmh=60.0,
                mass_kg=1500.0,
                distance_m=20.0,
                relative_speed_kmh=15.0,
                friction=noisy_mu,
                lead_deceleration_ms2=noisy_a,
                brake_light_detected=1
            )
            stages.append(dec.stage)

        # In a dangerous closing scenario, all noisy samples must consistently demand braking
        assert all(s in (BrakeStage.PARTIAL, BrakeStage.FULL) for s in stages)
