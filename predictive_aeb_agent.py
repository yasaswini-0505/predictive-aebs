"""
Predictive AEB Agent with Dynamic Temporal Feature Perception.

Implements PredictiveAEBAgent:
1. Ingests dynamic perception signals: lead deceleration, brake lights, relative acceleration, projected TTC.
2. Runs high-speed inference on aeb_predictive_rf_model.pkl via direct Cython tree evaluation.
3. Supports ML-Only mode and Predictive Hybrid (Predictive ML + Kinematic Safety Fallback).
4. Proactively initiates pre-braking / emergency braking based on predicted trajectory closure.
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

from braking_controller import BrakeStage
from aeb_agent import AEBDecision
from predictive_aeb_dataset_generator import compute_projected_ttc


class PredictiveAEBAgent:
    """
    Experimental Predictive AEB Agent utilizing dynamic perception signals.
    """
    is_predictive: bool = True

    FEATURE_NAMES = [
        "road_environment_Dry", "road_environment_Icy", "road_environment_Wet",
        "weather_condition_Clear", "weather_condition_Fog", "weather_condition_Rain",
        "vehicle_speed", "distance", "relative_speed", "lead_deceleration",
        "brake_light_detected", "relative_acceleration", "time_to_collision",
        "projected_ttc", "friction_estimate", "mass"
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        enable_physics_fallback: bool = True,
        brake_force_max: float = 12000.0,
        response_delay: float = 0.20,
        margin_critical: float = 1.0,
        margin_warning: float = 4.0,
        g: float = 9.81,
        hysteresis_hold: bool = True
    ):
        if model_path is None:
            model_path = str(PROJECT_ROOT / "aeb_predictive_rf_model.pkl")

        self.model_path = model_path
        self.enable_physics_fallback = enable_physics_fallback
        self.brake_force_max = float(brake_force_max)
        self.response_delay = float(response_delay)
        self.margin_critical = float(margin_critical)
        self.margin_warning = float(margin_warning)
        self.g = float(g)
        self.hysteresis_hold = hysteresis_hold

        self.latched_stage: BrakeStage = BrakeStage.OFF

        # Load predictive pipeline
        self.pipeline = joblib.load(self.model_path)
        self.clf = self.pipeline.named_steps["classifier"]
        self.trees = [t.tree_ for t in self.clf.estimators_]
        self._inv_n_trees = 1.0 / len(self.trees)

        # Mapping for fast one-hot indexing
        self.road_map = {"Dry": 0, "Icy": 1, "Wet": 2}
        self.weather_map = {"Clear": 0, "Fog": 1, "Rain": 2}
        self._buf = np.zeros((1, 16), dtype=np.float32)

    def compute_ttc(self, distance_m: float, relative_speed_kmh: float) -> float:
        v_rel_ms = relative_speed_kmh / 3.6
        if v_rel_ms > 0.05:
            return max(0.0, distance_m / v_rel_ms)
        return float("inf")

    def calculate_stopping_distances(
        self, mass_kg: float, friction: float, relative_speed_kmh: float
    ) -> Tuple[float, float]:
        v_rel_ms = relative_speed_kmh / 3.6
        if v_rel_ms <= 0.05:
            return 0.0, 0.0

        a_max = min(self.brake_force_max / max(1.0, float(mass_kg)), float(friction) * self.g)
        d_stop_full = (v_rel_ms * self.response_delay) + (v_rel_ms**2) / (2.0 * max(0.1, a_max))
        d_crit = d_stop_full + self.margin_critical

        a_part = 0.4 * a_max
        d_stop_part = (v_rel_ms * self.response_delay) + (v_rel_ms**2) / (2.0 * max(0.1, a_part))
        d_warn = d_stop_part + self.margin_warning

        return d_crit, d_warn

    def evaluate_state(
        self,
        vehicle_speed_kmh: float,
        mass_kg: float,
        distance_m: float,
        relative_speed_kmh: float,
        friction: float,
        road_environment: str = "Dry",
        weather_condition: str = "Clear",
        lead_deceleration_ms2: float = 0.0,
        brake_light_detected: int = 0
    ) -> AEBDecision:
        """
        Evaluates current dynamic state using predictive temporal perception.
        """
        if vehicle_speed_kmh <= 0.1 or distance_m <= 0:
            self.latched_stage = BrakeStage.OFF
            return AEBDecision(
                stage=BrakeStage.OFF,
                collision_probability=0.0,
                raw_prediction=0,
                ttc=float("inf"),
                reason="Vehicle stopped or collision occurred"
            )

        v_rel_ms = relative_speed_kmh / 3.6
        ttc = self.compute_ttc(distance_m, relative_speed_kmh)
        a_rel_ms2 = float(0.0 - lead_deceleration_ms2)
        proj_ttc = compute_projected_ttc(distance_m, v_rel_ms, a_rel_ms2)

        # 1. Populate inference buffer
        self._buf.fill(0.0)
        # Categorical one-hot (indices 0..5)
        r_idx = self.road_map.get(road_environment, 0)
        w_idx = self.weather_map.get(weather_condition, 0)
        self._buf[0, r_idx] = 1.0
        self._buf[0, 3 + w_idx] = 1.0

        # Numerical features (indices 6..15)
        self._buf[0, 6] = float(vehicle_speed_kmh)
        self._buf[0, 7] = max(0.1, float(distance_m))
        self._buf[0, 8] = max(0.0, float(relative_speed_kmh))
        self._buf[0, 9] = float(lead_deceleration_ms2)
        self._buf[0, 10] = float(brake_light_detected)
        self._buf[0, 11] = float(a_rel_ms2)
        self._buf[0, 12] = float(min(999.0, ttc))
        self._buf[0, 13] = float(min(999.0, proj_ttc))
        self._buf[0, 14] = float(friction)
        self._buf[0, 15] = float(mass_kg)

        # High-speed Cython tree evaluation
        prob_collision = sum(t.predict(self._buf)[0, 1] for t in self.trees) * self._inv_n_trees
        raw_pred = 1 if prob_collision >= 0.5 else 0

        stage = BrakeStage.OFF
        reason = "Normal driving"

        # Safe headway guard:
        if v_rel_ms <= 0.05 and lead_deceleration_ms2 >= -0.2:
            stage = BrakeStage.OFF
            reason = "Safe cruising: opening gap or matched speed"
        else:
            # 2. Predictive ML Control Logic
            # Emergency braking triggered if high predicted risk or acute projected closure
            if prob_collision >= 0.85 or (proj_ttc <= 1.8 and prob_collision >= 0.40):
                stage = BrakeStage.FULL
                reason = f"Predictive ML (FULL): P={prob_collision:.2f}, ProjTTC={proj_ttc:.2f}s, LeadAccel={lead_deceleration_ms2:.1f}m/s²"
            elif prob_collision >= 0.60 or (proj_ttc <= 2.6 and prob_collision >= 0.25) or (lead_deceleration_ms2 <= -3.0 and prob_collision >= 0.35):
                stage = BrakeStage.PARTIAL
                reason = f"Predictive ML (PARTIAL Pre-Brake): P={prob_collision:.2f}, ProjTTC={proj_ttc:.2f}s, LeadAccel={lead_deceleration_ms2:.1f}m/s²"

            # 3. Kinematic Safety Fallback (Guaranteed Physical Floor)
            if self.enable_physics_fallback:
                d_crit, d_warn = self.calculate_stopping_distances(mass_kg, friction, relative_speed_kmh)
                if distance_m <= d_crit or ttc <= 1.4:
                    if stage != BrakeStage.FULL:
                        stage = BrakeStage.FULL
                        reason = f"Physics safety fallback (FULL): gap={distance_m:.1f}m <= d_crit={d_crit:.1f}m, TTC={ttc:.2f}s (ML P={prob_collision:.2f})"
                elif distance_m <= d_warn or ttc <= 2.4:
                    if stage == BrakeStage.OFF:
                        stage = BrakeStage.PARTIAL
                        reason = f"Physics safety fallback (PARTIAL): gap={distance_m:.1f}m <= d_warn={d_warn:.1f}m, TTC={ttc:.2f}s (ML P={prob_collision:.2f})"

        # 4. Hysteresis Latching
        if self.hysteresis_hold:
            if self.latched_stage == BrakeStage.FULL and vehicle_speed_kmh > 0.1 and (relative_speed_kmh > 0.0 or lead_deceleration_ms2 < -0.5):
                stage = BrakeStage.FULL
                reason += " (Held by AEB hysteresis)"
            elif self.latched_stage == BrakeStage.PARTIAL and stage == BrakeStage.OFF and (ttc <= 2.4 or proj_ttc <= 2.6):
                stage = BrakeStage.PARTIAL
                reason += " (Held by partial hysteresis)"

        self.latched_stage = stage

        return AEBDecision(
            stage=stage,
            collision_probability=prob_collision,
            raw_prediction=raw_pred,
            ttc=ttc,
            reason=reason
        )

    def reset(self) -> None:
        self.latched_stage = BrakeStage.OFF
