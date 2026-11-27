"""
Tests for Dynamic Lead-Vehicle Kinematics and Acceleration/Deceleration Scenarios.

Verifies:
1. Constant-speed headway without nuisance braking.
2. Lead vehicle gradual deceleration and staged warning braking.
3. Sudden hard braking collision avoidance.
4. Emergency maximum braking mitigation.
5. Lead vehicle acceleration / opening gap stability.
6. Sub-step velocity clamping and non-negative speed enforcement.
7. Immediate collision termination at gap <= 0.
8. Heavy vehicle mass scaling and stopping-distance expansion.
9. Low-friction wet and icy road response.
"""

import pytest
from dynamic_simulation_engine import DynamicSimulationEngine, DynamicSimulationResult
from simulation_engine import SimulationOutcome
from aeb_agent import AEBAgent
from braking_controller import BrakeStage
from evaluate_hybrid_500_comparison import FastComparativeAEBAgent


class TestDynamicLeadBraking:
    """Test suite for DynamicSimulationEngine and dynamic maneuvers."""

    @pytest.fixture
    def hybrid_agent(self):
        """Creates a FastComparativeAEBAgent with hybrid physics fallback active."""
        return FastComparativeAEBAgent(enable_physics_fallback=True)

    @pytest.fixture
    def dynamic_engine(self, hybrid_agent):
        """Creates a DynamicSimulationEngine attached to the hybrid agent."""
        return DynamicSimulationEngine(time_step=0.01, max_duration=6.0, agent=hybrid_agent)

    def test_dynamic_constant_speed_headway(self, dynamic_engine):
        """
        Scenario: Ego and lead vehicle both cruise at 60 km/h with 25m headway.
        Expected: Headway is preserved, zero collisions, no nuisance braking.
        """
        res = dynamic_engine.run_scenario(
            initial_speed_kmh=60.0,
            mass_kg=1500.0,
            initial_distance_m=25.0,
            relative_speed_kmh=0.0,
            friction=0.85,
            lead_acceleration_ms2=0.0,
            lead_brake_start_time=0.0,
            lead_target_speed_kmh=60.0,
            aeb_enabled=True
        )

        assert not res.collision_occurred
        assert res.outcome in (SimulationOutcome.SAFE_STOP, SimulationOutcome.SAFE_TRANSIT)
        assert res.min_gap_m >= 24.9  # Headway maintained
        # Verify no unnecessary emergency braking occurred
        assert all(p.brake_stage == "OFF" for p in res.telemetry)

    def test_dynamic_lead_gradual_slowing(self, dynamic_engine):
        """
        Scenario: Lead vehicle gradually slows at -2.0 m/s^2 from 60 km/h to 20 km/h.
        Expected: AEB detects closing gap, engages staged braking, prevents collision.
        """
        res = dynamic_engine.run_scenario(
            initial_speed_kmh=60.0,
            mass_kg=1500.0,
            initial_distance_m=30.0,
            relative_speed_kmh=0.0,
            friction=0.80,
            lead_acceleration_ms2=-2.0,
            lead_brake_start_time=0.3,
            lead_target_speed_kmh=20.0,
            aeb_enabled=True
        )

        assert not res.collision_occurred
        assert res.min_gap_m > 2.0
        assert res.first_brake_time_s is not None
        # Verify braking was activated in stages
        stages_observed = {p.brake_stage for p in res.telemetry}
        assert "PARTIAL" in stages_observed or "FULL" in stages_observed

    def test_dynamic_lead_sudden_hard_braking(self, hybrid_agent):
        """
        Scenario: Urban 50 km/h, lead vehicle suddenly hard brakes at -6.0 m/s^2 to full stop.
        Compare: No AEB (crashes) vs Hybrid AEB (stops safely).
        """
        engine_none = DynamicSimulationEngine(time_step=0.01, max_duration=6.0, agent=None)
        res_none = engine_none.run_scenario(
            initial_speed_kmh=50.0,
            mass_kg=1500.0,
            initial_distance_m=20.0,
            relative_speed_kmh=0.0,
            friction=0.85,
            lead_acceleration_ms2=-6.0,
            lead_brake_start_time=0.4,
            lead_target_speed_kmh=0.0,
            aeb_enabled=False
        )

        # Baseline inattention causes collision
        assert res_none.collision_occurred
        assert res_none.impact_relative_speed_kmh > 20.0

        # Hybrid AEB must prevent collision
        engine_hyb = DynamicSimulationEngine(time_step=0.01, max_duration=6.0, agent=hybrid_agent)
        res_hyb = engine_hyb.run_scenario(
            initial_speed_kmh=50.0,
            mass_kg=1500.0,
            initial_distance_m=20.0,
            relative_speed_kmh=0.0,
            friction=0.85,
            lead_acceleration_ms2=-6.0,
            lead_brake_start_time=0.4,
            lead_target_speed_kmh=0.0,
            aeb_enabled=True
        )

        assert not res_hyb.collision_occurred
        assert res_hyb.outcome == SimulationOutcome.SAFE_STOP
        assert res_hyb.min_gap_m > 0.5

    def test_dynamic_lead_emergency_max_braking_mitigation(self, hybrid_agent):
        """
        Scenario: Extreme emergency braking (-8.5 m/s^2) with tight 12m headway at 65 km/h.
        Expected: Hybrid AEB significantly mitigates closing velocity compared to unassisted baseline.
        """
        engine_none = DynamicSimulationEngine(time_step=0.01, max_duration=5.0, agent=None)
        res_none = engine_none.run_scenario(
            initial_speed_kmh=65.0,
            mass_kg=1500.0,
            initial_distance_m=12.0,
            relative_speed_kmh=0.0,
            friction=0.85,
            lead_acceleration_ms2=-8.5,
            lead_brake_start_time=0.2,
            lead_target_speed_kmh=0.0,
            aeb_enabled=False
        )

        engine_hyb = DynamicSimulationEngine(time_step=0.01, max_duration=5.0, agent=hybrid_agent)
        res_hyb = engine_hyb.run_scenario(
            initial_speed_kmh=65.0,
            mass_kg=1500.0,
            initial_distance_m=12.0,
            relative_speed_kmh=0.0,
            friction=0.85,
            lead_acceleration_ms2=-8.5,
            lead_brake_start_time=0.2,
            lead_target_speed_kmh=0.0,
            aeb_enabled=True
        )

        assert res_none.collision_occurred
        # Even if a collision occurs due to tight headway, impact severity must be greatly mitigated
        if res_hyb.collision_occurred:
            assert res_hyb.impact_relative_speed_kmh < res_none.impact_relative_speed_kmh
        else:
            assert res_hyb.outcome == SimulationOutcome.SAFE_STOP

    def test_dynamic_lead_acceleration_opening_gap(self, dynamic_engine):
        """
        Scenario: Lead vehicle accelerates away at +2.5 m/s^2 to 90 km/h.
        Expected: Headway expands, closing speed is negative (opening), AEB remains OFF.
        """
        res = dynamic_engine.run_scenario(
            initial_speed_kmh=60.0,
            mass_kg=1500.0,
            initial_distance_m=20.0,
            relative_speed_kmh=0.0,
            friction=0.85,
            lead_acceleration_ms2=2.5,
            lead_brake_start_time=0.2,
            lead_target_speed_kmh=90.0,
            aeb_enabled=True
        )

        assert not res.collision_occurred
        assert res.final_gap_m > res.initial_gap_m
        assert all(p.brake_stage == "OFF" for p in res.telemetry)

    def test_dynamic_substep_zero_clamp_negative_speed_prevention(self, dynamic_engine):
        """
        Verifies that sub-step velocity clamping prevents any vehicle speed from ever becoming negative.
        """
        res = dynamic_engine.run_scenario(
            initial_speed_kmh=40.0,
            mass_kg=1300.0,
            initial_distance_m=25.0,
            relative_speed_kmh=0.0,
            friction=0.90,
            lead_acceleration_ms2=-7.0,
            lead_brake_start_time=0.1,
            lead_target_speed_kmh=0.0,
            aeb_enabled=True
        )

        for p in res.telemetry:
            assert p.vehicle_speed_kmh >= 0.0, f"Ego speed dropped below 0: {p.vehicle_speed_kmh}"
            assert p.obstacle_speed_kmh >= 0.0, f"Lead speed dropped below 0: {p.obstacle_speed_kmh}"

        assert res.final_speed_kmh >= 0.0
        assert res.min_gap_m >= 0.0

    def test_dynamic_collision_detection_zero_gap(self):
        """
        Verifies that collision detection terminates the simulation immediately upon gap <= 0.
        """
        engine = DynamicSimulationEngine(time_step=0.01, max_duration=5.0, agent=None)
        res = engine.run_scenario(
            initial_speed_kmh=60.0,
            mass_kg=1500.0,
            initial_distance_m=5.0,
            relative_speed_kmh=40.0,
            friction=0.85,
            lead_acceleration_ms2=0.0,
            aeb_enabled=False
        )

        assert res.collision_occurred
        assert res.outcome == SimulationOutcome.COLLISION
        assert res.final_gap_m == 0.0
        assert res.impact_speed_kmh > 0.0
        assert res.duration_s < 1.0  # Terminated quickly

    def test_dynamic_heavy_vehicle_stopping_distance(self, hybrid_agent):
        """
        Scenario: Same initial speed and lead deceleration, comparing light (1300 kg) vs heavy (2400 kg) ego.
        Expected: Heavy vehicle experiences smaller deceleration (a = F/m) resulting in larger stopping distance.
        """
        engine = DynamicSimulationEngine(time_step=0.01, max_duration=7.0, agent=hybrid_agent)

        res_light = engine.run_scenario(
            initial_speed_kmh=60.0,
            mass_kg=1300.0,
            initial_distance_m=35.0,
            relative_speed_kmh=0.0,
            friction=0.85,
            lead_acceleration_ms2=-5.0,
            lead_brake_start_time=0.3,
            lead_target_speed_kmh=0.0,
            aeb_enabled=True
        )

        res_heavy = engine.run_scenario(
            initial_speed_kmh=60.0,
            mass_kg=2400.0,
            initial_distance_m=35.0,
            relative_speed_kmh=0.0,
            friction=0.85,
            lead_acceleration_ms2=-5.0,
            lead_brake_start_time=0.3,
            lead_target_speed_kmh=0.0,
            aeb_enabled=True
        )

        assert not res_light.collision_occurred
        # Heavy vehicle stopping distance is longer, leaving a smaller safety margin
        assert res_heavy.stopping_distance_m > res_light.stopping_distance_m
        assert res_heavy.min_gap_m < res_light.min_gap_m

    def test_dynamic_low_friction_wet_road(self, dynamic_engine):
        """
        Scenario: Wet road (mu=0.45) with lead vehicle braking at -4.5 m/s^2.
        Expected: At 38m headway, physics fallback detects reduced traction and brings
        ego vehicle to a safe stop without collision.
        """
        res = dynamic_engine.run_scenario(
            initial_speed_kmh=50.0,
            mass_kg=1500.0,
            initial_distance_m=38.0,
            relative_speed_kmh=0.0,
            friction=0.45,
            road_environment="Wet",
            weather_condition="Rain",
            lead_acceleration_ms2=-4.5,
            lead_brake_start_time=0.4,
            lead_target_speed_kmh=0.0,
            aeb_enabled=True
        )

        assert not res.collision_occurred
        assert res.outcome in (SimulationOutcome.SAFE_STOP, SimulationOutcome.SAFE_TRANSIT)
        assert res.min_gap_m > 0.5

    def test_dynamic_telemetry_toggle_consistency(self, dynamic_engine):
        """
        Verifies that record_telemetry=True and record_telemetry=False produce
        identical kinematic outcomes and safety metrics.
        """
        kwargs = dict(
            initial_speed_kmh=60.0,
            mass_kg=1500.0,
            initial_distance_m=28.0,
            relative_speed_kmh=0.0,
            friction=0.85,
            lead_acceleration_ms2=-5.0,
            lead_brake_start_time=0.3,
            lead_target_speed_kmh=0.0,
            aeb_enabled=True
        )

        res_with = dynamic_engine.run_scenario(**kwargs, record_telemetry=True)
        res_without = dynamic_engine.run_scenario(**kwargs, record_telemetry=False)

        assert res_with.collision_occurred == res_without.collision_occurred
        assert res_with.outcome == res_without.outcome
        assert abs(res_with.min_gap_m - res_without.min_gap_m) < 1e-4
        assert abs(res_with.final_gap_m - res_without.final_gap_m) < 1e-4
        assert abs(res_with.stopping_distance_m - res_without.stopping_distance_m) < 1e-4
        assert len(res_with.telemetry) > 0
        assert len(res_without.telemetry) == 0

    def test_dynamic_500_suite_generation_integrity(self):
        """
        Verifies that the 500 dynamic scenario generator produces exact counts,
        reproducible IDs, and valid physical constraints across all cohorts.
        """
        from experiment_dynamic_500 import generate_500_dynamic_scenarios
        suite1 = generate_500_dynamic_scenarios(seed=42)
        suite2 = generate_500_dynamic_scenarios(seed=42)

        assert len(suite1) == 500
        assert len(suite2) == 500
        for s1, s2 in zip(suite1, suite2):
            assert s1.scenario_id == s2.scenario_id
            assert s1.cohort == s2.cohort
            assert s1.v_ego_kmh == s2.v_ego_kmh
            assert s1.gap_m == s2.gap_m
            assert s1.lead_accel_ms2 == s2.lead_accel_ms2

        # Check cohort distribution
        cohorts = {s.cohort for s in suite1}
        assert len(cohorts) == 6
        assert sum(1 for s in suite1 if s.is_safe_following) == 80
        assert sum(1 for s in suite1 if s.is_physically_unavoidable) == 80
