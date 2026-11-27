"""
Loads the ORIGINAL BTP decision logic. Nothing here re-implements a model or a threshold.

  static_ml     -> experiment_dynamic_500.UltraFastComparativeAEBAgent(enable_physics_fallback=False)
  static_hybrid -> experiment_dynamic_500.UltraFastComparativeAEBAgent(enable_physics_fallback=True)
                   (the initial aeb_random_forest_model.pkl; same agent class as the original 500-scenario run)
  hybrid  -> predictive_aeb_agent.PredictiveAEBAgent(enable_physics_fallback=True)
  ml      -> predictive_aeb_agent.PredictiveAEBAgent(enable_physics_fallback=False)
  physics -> experiment_close_obstacle_baseline.KinematicAEBAgent()
  no_aeb  -> None
"""
import config  # noqa: F401  (adds the BTP folder to sys.path)
from config import PREDICTIVE_MODEL, STATIC_MODEL, AEB_RESPONSE_DELAY, BTP_BRAKE_FORCE_N

from braking_controller import BrakingController, BrakeStage          # noqa: E402
from predictive_aeb_agent import PredictiveAEBAgent                     # noqa: E402
from predictive_aeb_dataset_generator import compute_projected_ttc      # noqa: E402
from experiment_close_obstacle_baseline import KinematicAEBAgent        # noqa: E402
from experiment_dynamic_500 import UltraFastComparativeAEBAgent          # noqa: E402

__all__ = ["make_agent", "BrakingController", "BrakeStage", "compute_projected_ttc", "model_feature_names"]

_model_cache = {}


def make_agent(mode: str):
    if mode == "no_aeb":
        return None
    if mode in ("static_ml", "static_hybrid"):
        return UltraFastComparativeAEBAgent(model_path=str(STATIC_MODEL),
                                            enable_physics_fallback=(mode == "static_hybrid"),
                                            brake_force_max=BTP_BRAKE_FORCE_N, response_delay=AEB_RESPONSE_DELAY)
    if mode == "physics":
        return KinematicAEBAgent(brake_force_max=BTP_BRAKE_FORCE_N, latency=AEB_RESPONSE_DELAY)
    if mode in ("ml", "hybrid"):
        return PredictiveAEBAgent(model_path=str(PREDICTIVE_MODEL),
                                  enable_physics_fallback=(mode == "hybrid"),
                                  brake_force_max=BTP_BRAKE_FORCE_N,
                                  response_delay=AEB_RESPONSE_DELAY)
    raise ValueError(f"Unknown mode {mode!r}")


def model_feature_names():
    """Feature names exactly as stored inside the saved pipeline."""
    import joblib
    if "p" not in _model_cache:
        _model_cache["p"] = joblib.load(PREDICTIVE_MODEL)
    p = _model_cache["p"]
    return list(p.feature_names_in_), list(p.named_steps["preprocessor"].get_feature_names_out())
