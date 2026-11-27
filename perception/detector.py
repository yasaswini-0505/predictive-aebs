"""
Lead-vehicle detection and collision metrics from MetaDrive ground-truth vehicle states.

This is NOT a radar/camera model: positions and velocities are read from the simulator.
Lead acceleration is ESTIMATED from the measured lead speed (finite difference over a short
window), so it lags slightly like a real tracker would. The brake-light signal is derived
from that estimate with the same rule as the original BTP engine (lead accel < -0.8 m/s^2).
"""
import math
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

import config as C
from ml.btp_agents import compute_projected_ttc

MIN_CLOSING_SPEED = 0.05   # m/s; below this TTC is reported as None ("not closing"), same threshold as BTP agents
ACCEL_WINDOW = 5           # steps (0.1 s) used for acceleration estimates
BRAKE_LIGHT_THRESHOLD = -0.8


@dataclass
class LeadState:
    t: float
    lead_found: bool
    ego_speed_mps: float
    ego_speed_kmh: float
    ego_accel_mps2: float
    lead_speed_mps: Optional[float] = None
    lead_speed_kmh: Optional[float] = None
    distance_m: Optional[float] = None            # bumper to bumper
    lateral_offset_m: Optional[float] = None
    relative_speed_mps: Optional[float] = None    # + = closing
    relative_speed_kmh: Optional[float] = None
    lead_accel_mps2: Optional[float] = None       # estimated; < 0 = braking
    relative_accel_mps2: Optional[float] = None   # measured a_ego - a_lead
    ttc_s: Optional[float] = None                 # None = not closing
    projected_ttc_s: Optional[float] = None       # BTP formula, None = no collision predicted
    brake_light_detected: int = 0
    lead_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def time_to_collision(distance_m: float, closing_speed_mps: float) -> Optional[float]:
    if distance_m is None or closing_speed_mps is None or closing_speed_mps <= MIN_CLOSING_SPEED:
        return None
    return max(0.0, distance_m / closing_speed_mps)


def projected_ttc(distance_m, closing_speed_mps, lead_accel_mps2) -> Optional[float]:
    """Uses the project's own compute_projected_ttc with the same inputs the agent uses (a_rel = 0 - a_lead)."""
    v = compute_projected_ttc(distance_m, closing_speed_mps, 0.0 - lead_accel_mps2)
    return None if v >= 999.0 else float(v)


def _speed_along(vehicle, unit_heading) -> float:
    vx, vy = vehicle.velocity
    return float(vx * unit_heading[0] + vy * unit_heading[1])


class LeadDetector:
    """Finds the nearest vehicle ahead in the ego lane, in the ego car's heading frame."""

    def __init__(self, max_range_m=150.0, max_heading_diff_rad=0.5):
        self.max_range = max_range_m
        self.max_heading_diff = max_heading_diff_rad
        self.reset()

    def reset(self):
        self._lead_speeds = deque(maxlen=ACCEL_WINDOW + 1)
        self._ego_speeds = deque(maxlen=ACCEL_WINDOW + 1)
        self._lead_id = None

    @staticmethod
    def _accel(hist):
        if len(hist) < 2:
            return 0.0
        return (hist[-1] - hist[0]) / ((len(hist) - 1) * C.STEP_DT)

    def candidates(self, ego, vehicles):
        """All vehicles ahead in the same lane, nearest first: (gap_m, lateral_m, vehicle)."""
        h = np.asarray(ego.heading, dtype=float)
        h /= (np.linalg.norm(h) + 1e-9)
        ex, ey = ego.position
        half_width = ego.lane.width / 2.0 if ego.lane is not None else 1.75
        out = []
        for v in vehicles:
            if v is ego or getattr(v, "id", None) == ego.id:
                continue
            rx, ry = v.position[0] - ex, v.position[1] - ey
            lon = rx * h[0] + ry * h[1]
            lat = -rx * h[1] + ry * h[0]
            if lon <= 0 or lon > self.max_range:            # behind or too far
                continue
            if abs(lat) >= half_width:                       # adjacent lane
                continue
            dh = math.atan2(math.sin(v.heading_theta - ego.heading_theta), math.cos(v.heading_theta - ego.heading_theta))
            if abs(dh) > self.max_heading_diff:              # oncoming / crossing
                continue
            if ego.lane_index is not None and v.lane_index is not None and ego.lane_index[-1] != v.lane_index[-1]:
                continue                                     # different lane number
            gap = lon - (ego.LENGTH + v.LENGTH) / 2.0
            out.append((gap, lat, v))
        out.sort(key=lambda c: c[0])
        return out

    def update(self, t, ego, vehicles) -> LeadState:
        h = np.asarray(ego.heading, dtype=float)
        h /= (np.linalg.norm(h) + 1e-9)
        v_ego = _speed_along(ego, h)
        self._ego_speeds.append(v_ego)
        a_ego = self._accel(self._ego_speeds)
        cands = self.candidates(ego, vehicles)
        if not cands:
            self._lead_speeds.clear(); self._lead_id = None
            return LeadState(t=t, lead_found=False, ego_speed_mps=v_ego, ego_speed_kmh=v_ego * 3.6, ego_accel_mps2=a_ego)
        gap, lat, lead = cands[0]
        if lead.id != self._lead_id:                         # new target: restart its history
            self._lead_speeds.clear(); self._lead_id = lead.id
        v_lead = _speed_along(lead, h)
        self._lead_speeds.append(v_lead)
        a_lead = self._accel(self._lead_speeds)
        closing = v_ego - v_lead
        return LeadState(
            t=t, lead_found=True, ego_speed_mps=v_ego, ego_speed_kmh=v_ego * 3.6, ego_accel_mps2=a_ego,
            lead_speed_mps=v_lead, lead_speed_kmh=v_lead * 3.6, distance_m=gap, lateral_offset_m=lat,
            relative_speed_mps=closing, relative_speed_kmh=closing * 3.6,
            lead_accel_mps2=a_lead, relative_accel_mps2=a_ego - a_lead,
            ttc_s=time_to_collision(gap, closing), projected_ttc_s=projected_ttc(max(gap, 0.0), closing, a_lead),
            brake_light_detected=int(a_lead < BRAKE_LIGHT_THRESHOLD and v_lead > 0.1), lead_id=lead.id)
