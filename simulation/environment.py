"""
Creates a deterministic MetaDrive world with exactly two cars and no random traffic.

Supported MetaDrive 0.4.3 APIs used (checked in the installed source):
  * VaryingDynamicsEnv + random_dynamics with min == max   -> fixed ego mass, tyre friction, brake force
  * engine.spawn_object(LVehicle, vehicle_config=...)      -> the lead car
  * BaseObject.set_velocity(direction, value)               -> scripted lead speed every physics step
  * engine.clear_objects([id], force_destroy=True)          -> remove the lead before the next scenario
"""
import json

from metadrive.envs.varying_dynamics_env import VaryingDynamicsEnv
from metadrive.component.vehicle.vehicle_type import LVehicle

LEAD_CLASS = LVehicle   # a different car model from the ego car, so the two are easy to tell apart
from metadrive.component.pgblock.first_block import FirstPGBlock

import config as C

SPAWN_LANE = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, C.LANE_INDEX)


def wheel_friction_for(surface: str) -> float:
    """MetaDrive tyre friction for a surface, from the saved calibration (dry uses MetaDrive's 0.9)."""
    if C.CALIBRATION_FILE.exists():
        data = json.loads(C.CALIBRATION_FILE.read_text())
        wf = data.get("wheel_friction", {}).get(surface)
        if wf is not None:
            return float(wf)
    if surface == "Dry":
        return 0.9
    raise RuntimeError(f"No tyre friction calibrated for {surface}. Run: python main.py calibrate")


def make_env(surface="Dry", mass_kg=1500.0, render=False, wheel_friction=None, window=(1280, 720)):
    wf = wheel_friction_for(surface) if wheel_friction is None else wheel_friction
    fixed = lambda v: (v, v)
    cfg = dict(
        use_render=render,
        num_scenarios=1, start_seed=0, map=C.MAP,
        traffic_density=0.0, accident_prob=0.0,
        decision_repeat=C.DECISION_REPEAT, physics_world_step_size=C.PHYSICS_DT,
        horizon=None, log_level=50,
        random_spawn_lane_index=False,
        crash_vehicle_done=False, out_of_road_done=False,   # this package decides when a run ends
        random_dynamics=dict(max_engine_force=fixed(C.MAX_ENGINE_FORCE), max_brake_force=fixed(C.MAX_BRAKE_FORCE),
                             wheel_friction=fixed(wf), max_steering=fixed(C.MAX_STEERING), mass=fixed(float(mass_kg))),
        vehicle_config=dict(spawn_lane_index=SPAWN_LANE, spawn_longitude=C.EGO_SPAWN_LONGITUDE,
                            show_navi_mark=False, show_dest_mark=False, show_navigation_arrow=False),
    )
    if render:
        cfg.update(window_size=window, show_interface=False, force_render_fps=int(1 / C.STEP_DT),
                   camera_height=3.0, camera_dist=11.0)   # a little higher/further back so the lead car is in view
    env = VaryingDynamicsEnv(cfg)
    env.aeb_surface, env.aeb_mass, env.aeb_wheel_friction = surface, float(mass_kg), wf
    return env


class LeadVehicle:
    """A lead car whose speed follows a script: cruise, then accelerate/brake at a fixed rate until a target speed."""

    def __init__(self, env, ego, scenario: C.Scenario):
        self.env, self.sc = env, scenario
        lane = ego.lane
        ego_long, _ = lane.local_coordinates(ego.position)
        cfg = ego.config.copy()
        length = LEAD_CLASS.DEFAULT_LENGTH
        centre_long = ego_long + ego.LENGTH / 2 + scenario.gap_m + length / 2
        cfg.update(dict(spawn_lane_index=ego.lane_index, spawn_longitude=centre_long, spawn_lateral=0.0,
                        spawn_velocity=None, use_special_color=False))
        self.vehicle = env.engine.spawn_object(LEAD_CLASS, vehicle_config=cfg, random_seed=0)
        self.speed = scenario.lead_speed_kmh / 3.6     # commanded speed (m/s)
        self.accel = 0.0                               # commanded acceleration this step (m/s^2)
        self.vehicle.set_velocity(self.vehicle.heading, self.speed)

    def command(self, t: float):
        """Called once per step before env.step(). Returns the commanded acceleration."""
        sc = self.sc
        target = sc.lead_target_speed_kmh / 3.6
        a = 0.0
        if t >= sc.lead_event_time_s and sc.lead_accel_ms2 != 0.0:
            if sc.lead_accel_ms2 < 0 and self.speed > target:
                a = sc.lead_accel_ms2
            elif sc.lead_accel_ms2 > 0 and self.speed < target:
                a = sc.lead_accel_ms2
        new = self.speed + a * C.STEP_DT
        if a < 0:
            new = max(new, target, 0.0)
        elif a > 0:
            new = min(new, target)
        self.accel = (new - self.speed) / C.STEP_DT
        self.speed = new
        self.vehicle.set_velocity(self.vehicle.heading, self.speed)
        return self.accel

    def remove(self):
        try:
            self.env.engine.clear_objects([self.vehicle.id], force_destroy=True)
        except Exception:
            pass
