"""Fast tests that do not start MetaDrive."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

import config as C
from perception.detector import LeadDetector, time_to_collision, projected_ttc
from control.brake_mapper import BrakeMapper
from control.aeb_controller import AEBController, CruiseDriver
from ml.btp_agents import make_agent, model_feature_names, BrakeStage


class FakeLane:
    width = 3.5


def car(vid, x, y, vx, vy=0.0, theta=None, lane=0, length=4.5):
    theta = math.atan2(vy, vx) if theta is None else theta
    return SimpleNamespace(id=vid, position=(x, y), velocity=(vx, vy), heading=(math.cos(theta), math.sin(theta)),
                           heading_theta=theta, lane_index=(">", ">>", lane), lane=FakeLane(), LENGTH=length)


def test_ttc_rules():
    assert time_to_collision(20.0, 5.0) == pytest.approx(4.0)
    assert time_to_collision(20.0, 0.0) is None          # zero closing speed
    assert time_to_collision(20.0, -3.0) is None         # opening gap
    assert time_to_collision(20.0, 0.04) is None         # below threshold, avoids huge TTC
    assert projected_ttc(20.0, 5.0, -4.0) < 4.0          # lead braking shortens projected TTC
    assert projected_ttc(20.0, 0.0, 0.0) is None


def test_nearest_same_lane_lead_selected():
    ego = car("ego", 0, 0, 15)
    vehicles = [car("behind", -20, 0, 15), car("adjacent", 10, 3.5, 10, lane=1),
                car("far", 60, 0, 10), car("near", 30, 0.3, 10), car("oncoming", 20, 0, -10)]
    s = LeadDetector().update(0.0, ego, vehicles)
    assert s.lead_found and s.lead_id == "near"
    assert s.distance_m == pytest.approx(30 - 4.5)        # bumper to bumper
    assert s.relative_speed_mps == pytest.approx(5.0)
    assert s.ttc_s == pytest.approx(25.5 / 5.0)


def test_heading_aware_frame_not_x_axis():
    """Road pointing along +y: x-difference alone would give a wrong answer."""
    th = math.pi / 2
    ego = car("ego", 0, 0, 0, 15, theta=th)
    lead = car("lead", 0.2, 25, 0, 10, theta=th)
    side = car("side", 3.6, 10, 0, 10, theta=th, lane=1)
    s = LeadDetector().update(0.0, ego, [side, lead])
    assert s.lead_id == "lead" and s.distance_m == pytest.approx(25 - 4.5)
    assert s.relative_speed_mps == pytest.approx(5.0)


def test_no_lead_and_opening_gap():
    d = LeadDetector()
    assert not d.update(0.0, car("ego", 0, 0, 10), []).lead_found
    s = d.update(0.02, car("ego", 0, 0, 10), [car("lead", 30, 0, 15)])
    assert s.relative_speed_mps < 0 and s.ttc_s is None


def test_lead_acceleration_estimate_and_brake_light():
    d = LeadDetector()
    for i in range(6):
        s = d.update(i * C.STEP_DT, car("ego", 0, 0, 15), [car("lead", 30, 0, 10 - 6 * i * C.STEP_DT)])
    assert s.lead_accel_mps2 == pytest.approx(-6.0, abs=1e-6)
    assert s.brake_light_detected == 1


def test_vehicle_replaced_resets_history():
    d = LeadDetector()
    d.update(0.0, car("ego", 0, 0, 15), [car("a", 30, 0, 10)])
    s = d.update(0.02, car("ego", 0, 0, 15), [car("b", 20, 0, 0)])   # different car appears
    assert s.lead_id == "b" and s.lead_accel_mps2 == 0.0


def test_brake_mapper_inverts_measured_table():
    m = BrakeMapper([0.1, 0.2, 0.4, 1.0], [2.0, 4.0, 8.0, 12.0])
    assert m.action_for(0.0) == 0.0
    assert m.action_for(4.0) == pytest.approx(-0.2)
    assert m.action_for(6.0) == pytest.approx(-0.3)
    assert m.action_for(50.0) == -1.0
    assert all(m.action_for(x) <= 0 for x in np.linspace(0, 20, 50))


def test_model_feature_order_matches_agent():
    raw, enc = model_feature_names()
    assert raw == ['road_environment', 'weather_condition', 'vehicle_speed', 'distance', 'relative_speed',
                   'lead_deceleration', 'brake_light_detected', 'relative_acceleration', 'time_to_collision',
                   'projected_ttc', 'friction_estimate', 'mass']
    assert [n.split("__", 1)[1] for n in enc] == make_agent("hybrid").FEATURE_NAMES


def test_agents_are_the_original_classes():
    assert type(make_agent("hybrid")).__module__ == "predictive_aeb_agent"
    assert make_agent("hybrid").enable_physics_fallback and not make_agent("ml").enable_physics_fallback
    assert type(make_agent("physics")).__name__ == "KinematicAEBAgent"
    static_ml, static_hyb = make_agent("static_ml"), make_agent("static_hybrid")
    assert type(static_ml).__name__ == "UltraFastComparativeAEBAgent"          # agent from the original 500-run
    assert static_ml.model_path.endswith("aeb_random_forest_model.pkl")      # the initial Random Forest
    assert not static_ml.enable_physics_fallback and static_hyb.enable_physics_fallback
    assert make_agent("no_aeb") is None


def test_ml_prediction_danger_vs_safe():
    agent = make_agent("ml")
    danger = agent.evaluate_state(vehicle_speed_kmh=60, mass_kg=1500, distance_m=12, relative_speed_kmh=40,
                                  friction=0.85, lead_deceleration_ms2=-6, brake_light_detected=1)
    agent.reset()
    safe = agent.evaluate_state(vehicle_speed_kmh=60, mass_kg=1500, distance_m=60, relative_speed_kmh=0,
                                friction=0.85, lead_deceleration_ms2=0, brake_light_detected=0)
    assert danger.collision_probability > 0.85 and danger.stage == BrakeStage.FULL
    assert safe.collision_probability < 0.2 and safe.stage == BrakeStage.OFF


def test_controller_response_delay_and_stage_mapping():
    mapper = BrakeMapper([0.1, 0.2, 0.4, 1.0], [2.0, 4.0, 8.0, 12.0])
    ctrl = AEBController("hybrid", 1500, "Dry", mapper, set_speed_mps=16.7)
    from perception.detector import LeadState
    danger = LeadState(t=0, lead_found=True, ego_speed_mps=16.7, ego_speed_kmh=60, ego_accel_mps2=0,
                       lead_speed_mps=5, lead_speed_kmh=18, distance_m=12, relative_speed_mps=11.7,
                       relative_speed_kmh=42, lead_accel_mps2=-6, relative_accel_mps2=6, brake_light_detected=1)
    out0 = ctrl.step(0.0, danger)
    assert out0.stage == "FULL" and out0.decel_request == 0.0 and out0.action[1] >= 0   # within 0.20 s delay
    out1 = ctrl.step(0.10, danger)
    assert out1.decel_request == 0.0
    out2 = ctrl.step(0.20, danger)
    assert out2.decel_request == pytest.approx(8.0)                                      # min(12000/1500, 0.85 g)
    assert out2.action[1] == pytest.approx(-0.4)                                          # mapped through table
    assert out2.throttle == 0.0                                                           # never throttle + brake


def test_wet_surface_caps_deceleration():
    mapper = BrakeMapper([0.1, 0.4, 1.0], [2.0, 4.9, 4.9])
    ctrl = AEBController("physics", 1500, "Wet", mapper, set_speed_mps=14)
    from perception.detector import LeadState
    s = LeadState(t=0, lead_found=True, ego_speed_mps=14, ego_speed_kmh=50, ego_accel_mps2=0, lead_speed_mps=0,
                  lead_speed_kmh=0, distance_m=8, relative_speed_mps=14, relative_speed_kmh=50, lead_accel_mps2=0,
                  relative_accel_mps2=0)
    ctrl.step(0.0, s)
    assert ctrl.step(0.2, s).decel_request == pytest.approx(0.5 * 9.81)


def test_cruise_driver_never_brakes():
    d = CruiseDriver(15.0)
    assert d.action(20.0) == 0.0 and 0 < d.action(15.0) <= 1.0
