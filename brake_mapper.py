"""
Converts a deceleration request (m/s^2) from the BTP BrakingController into MetaDrive's
throttle_brake action in [-1, 1].

MetaDrive 0.4.3 (base_vehicle._apply_throttle_brake): a negative value applies
|value| * max_brake_force to every wheel. The resulting deceleration depends on mass and
tyre friction, so it is MEASURED (calibrate.py) instead of assuming 0.4 stage -> -0.4 action.
"""
import numpy as np


class BrakeMapper:
    def __init__(self, commands, decels):
        c = np.asarray(commands, dtype=float)       # brake magnitudes, e.g. 0.05 ... 1.0
        d = np.asarray(decels, dtype=float)         # measured decelerations (m/s^2)
        order = np.argsort(c)
        c, d = c[order], np.maximum.accumulate(np.maximum(d[order], 0.0))   # enforce monotone
        self.commands = np.concatenate([[0.0], c])
        self.decels = np.concatenate([[0.0], d])
        self.max_decel = float(self.decels[-1])

    def action_for(self, decel_request: float) -> float:
        """MetaDrive throttle_brake value (<= 0) that produces the requested deceleration."""
        if decel_request <= 0:
            return 0.0
        if decel_request >= self.max_decel:
            return -1.0
        # first command whose measured deceleration reaches the request (interpolated)
        cmd = float(np.interp(decel_request, self.decels, self.commands))
        return -min(1.0, max(0.0, cmd))

    def to_dict(self):
        return dict(commands=self.commands[1:].tolist(), decels=self.decels[1:].tolist(), max_decel=self.max_decel)
