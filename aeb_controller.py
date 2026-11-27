"""
The per-step AEB loop. It only connects existing pieces:

  LeadState -> BTP agent.evaluate_state(...) -> BrakeStage
            -> BTP BrakingController.set_target_stage / compute_deceleration (0.20 s delay)
            -> BrakeMapper -> MetaDrive throttle_brake

When no braking is being applied the 'driver' holds the starting speed (an inattentive
driver who never brakes, the same assumption as the original No-AEB baseline).
"""
from dataclasses import dataclass
from typing import Optional

import config as C
from ml.btp_agents import make_agent, BrakingController, BrakeStage
from perception.detector import LeadState


class CruiseDriver:
    """Holds a set speed with throttle only (never brakes)."""

    def __init__(self, set_speed_mps, gain=0.6, hold=0.12):
        self.set_speed, self.gain, self.hold = set_speed_mps, gain, hold

    def action(self, speed_mps):
        return float(min(1.0, max(0.0, self.hold + self.gain * (self.set_speed - speed_mps))))


@dataclass
class ControlOutput:
    action: list
    stage: str
    decel_request: float
    brake_command: float          # 0..1 fraction of MetaDrive brake
    throttle: float
    probability: Optional[float]
    reason: str


class AEBController:
    def __init__(self, mode, mass_kg, surface, mapper, set_speed_mps):
        self.mode = mode
        self.mass, self.surface = mass_kg, surface
        self.mu = C.SURFACES[surface]["mu"]
        self.weather = C.SURFACES[surface]["weather"]
        self.agent = make_agent(mode)
        self.brakes = BrakingController(mass=mass_kg, friction=self.mu, max_brake_force=C.BTP_BRAKE_FORCE_N,
                                        response_delay=C.AEB_RESPONSE_DELAY)
        self.mapper = mapper
        self.driver = CruiseDriver(set_speed_mps)
        self.first_brake_time = None
        self.reset()

    def reset(self):
        if self.agent is not None:
            self.agent.reset() if hasattr(self.agent, "reset") else setattr(self.agent, "latched_stage", BrakeStage.OFF)
        self.brakes.reset()
        self.first_brake_time = None

    def _decide(self, s: LeadState):
        if self.agent is None or not s.lead_found:
            return BrakeStage.OFF, None, "No AEB" if self.agent is None else "No lead vehicle"
        kw = dict(vehicle_speed_kmh=s.ego_speed_kmh, mass_kg=self.mass, distance_m=max(0.0, s.distance_m),
                  relative_speed_kmh=s.relative_speed_kmh, friction=self.mu,
                  road_environment=self.surface, weather_condition=self.weather)
        if self.mode in ("ml", "hybrid"):
            kw.update(lead_deceleration_ms2=s.lead_accel_mps2, brake_light_detected=s.brake_light_detected)
        d = self.agent.evaluate_state(**kw)
        return d.stage, (d.collision_probability if self.mode in C.ML_MODES else None), d.reason

    def step(self, t, s: LeadState) -> ControlOutput:
        stage, prob, reason = self._decide(s)
        self.brakes.set_target_stage(stage, current_time=t)
        decel = self.brakes.compute_deceleration(t, s.ego_speed_mps)
        if stage != BrakeStage.OFF and self.first_brake_time is None:
            self.first_brake_time = t
        if decel > 0:
            cmd = self.mapper.action_for(decel)
            return ControlOutput([0.0, cmd], stage.name, decel, -cmd, 0.0, prob, reason)
        throttle = self.driver.action(s.ego_speed_mps)
        return ControlOutput([0.0, throttle], stage.name, 0.0, 0.0, throttle, prob, reason)
