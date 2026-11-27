"""
Settings for the MetaDrive AEB integration.

Units used everywhere in this package: metres, seconds, m/s, m/s^2, kg.
km/h appears only where the original BTP agents require it (their inputs end in _kmh).
Sign conventions:
  * relative (closing) speed = ego speed - lead speed; positive means the gap is shrinking.
  * lead acceleration is negative when the lead car brakes (e.g. -6 m/s^2).
"""
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

PACKAGE_DIR = Path(__file__).resolve().parent
BTP_ROOT = PACKAGE_DIR.parent                      # /Users/udaykiran/Desktop/BTP
RESULTS_DIR = PACKAGE_DIR / "results"
CALIBRATION_FILE = PACKAGE_DIR / "control" / "brake_calibration.json"
PREDICTIVE_MODEL = BTP_ROOT / "aeb_predictive_rf_model.pkl"
STATIC_MODEL = BTP_ROOT / "aeb_random_forest_model.pkl"      # the initial Random Forest

# Make the original BTP modules importable without copying them.
if str(BTP_ROOT) not in sys.path:
    sys.path.insert(0, str(BTP_ROOT))
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

# ---- simulation timing -------------------------------------------------------
PHYSICS_DT = 0.02          # MetaDrive physics step (s)
DECISION_REPEAT = 1        # one physics step per env.step -> AEB runs at 50 Hz
STEP_DT = PHYSICS_DT * DECISION_REPEAT
MAP = "SSSSSS"             # straight blocks only (about 470 m) so the experiment is not affected by curves
EGO_SPAWN_LONGITUDE = 5.0  # metres along the first lane
LANE_INDEX = 0             # both cars use lane 0 of the first road

# ---- vehicle dynamics (fixed so every run is repeatable) --------------------
MAX_ENGINE_FORCE = 800.0   # MetaDrive default-vehicle range is 750-850
MAX_BRAKE_FORCE = 150.0    # MetaDrive default-vehicle range is 80-180
MAX_STEERING = 40.0
BTP_BRAKE_FORCE_N = 12000.0  # BrakingController / agents' brake force (original project value)
AEB_RESPONSE_DELAY = 0.20    # s, same as the original BrakingController default
METADRIVE_SPEED_LIMIT_KMH = 80.0  # MetaDrive cuts engine force above this; keep ego speeds <= 80 km/h

# Road surfaces: friction mu given to the BTP agents/controller, the weather label the
# model was trained with, and MetaDrive's tyre "wheel_friction" chosen by calibration so
# that the car's maximum braking in MetaDrive is close to mu * g (see control/calibrate.py).
SURFACES = {
    "Dry": dict(mu=0.85, weather="Clear"),
    "Wet": dict(mu=0.50, weather="Rain"),
    "Icy": dict(mu=0.20, weather="Fog"),
}

# Order = the project's development story (initial ML -> physics -> predictive ML).
MODES = ["no_aeb", "static_ml", "static_hybrid", "physics", "ml", "hybrid"]
MODE_LABELS = {
    "no_aeb": "No AEB",
    "static_ml": "Static ML-only AEB",
    "static_hybrid": "Static Hybrid AEB",
    "physics": "Physics-only AEB",
    "ml": "Predictive ML-only AEB",
    "hybrid": "Predictive Hybrid AEB",
}
def short_label(mode):
    """'Physics-only' instead of 'Physics-only AEB', but keep 'No AEB' readable."""
    return MODE_LABELS[mode] if mode == "no_aeb" else MODE_LABELS[mode].replace(" AEB", "")


ML_MODES = ("static_ml", "static_hybrid", "ml", "hybrid")   # modes that output a model probability

MAX_DURATION = 12.0        # s per scenario


@dataclass
class Scenario:
    """One controlled two-car experiment. Speeds in km/h for readability, converted internally."""
    scenario_id: str
    description: str
    ego_speed_kmh: float
    lead_speed_kmh: float
    gap_m: float                       # bumper-to-bumper distance at t = 0
    lead_accel_ms2: float = 0.0        # applied from lead_event_time_s; < 0 brakes, > 0 accelerates
    lead_event_time_s: float = 2.0
    lead_target_speed_kmh: float = 0.0 # lead stops changing speed when it reaches this
    surface: str = "Dry"
    mass_kg: float = 1500.0
    category: str = "emergency"        # "emergency", "safe", ...
    expect_no_braking: bool = False    # True for safe scenarios (false-braking check)

    def to_dict(self):
        return asdict(self)
