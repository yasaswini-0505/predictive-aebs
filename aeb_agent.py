from dataclasses import dataclass
from typing import Optional, Tuple
import joblib
import pandas as pd

from braking_controller import BrakeStage


@dataclass
class AEBDecision:
    """Telemetry structure representing an AEB decision cycle."""
    stage: BrakeStage
    collision_probability: float
    raw_prediction: int
    ttc: float  # Time-to-Collision in seconds
    reason: str


class AEBAgent:
    """
    ML-to-Simulation Interface for Autonomous Emergency Braking.
    
    Bridges the real-time physical simulation state with the trained Machine Learning
    pipeline (`aeb_random_forest_model.pkl`), extracting feature vectors, performing
    probabilistic collision inference, evaluating kinematic TTC, and deciding multi-stage
    braking commands.
    
    Attributes:
        model_path (str): Path to trained model pickle.
        partial_risk_threshold (float): Probability threshold to trigger partial / warning brake.
        full_risk_threshold (float): Probability threshold to trigger full emergency brake.
        ttc_critical (float): Time-to-Collision threshold for emergency braking in seconds.
        ttc_warning (float): Time-to-Collision threshold for partial braking in seconds.
        hysteresis_hold (bool): Maintain braking once triggered until vehicle stops or risk clears.
    """

    FEATURE_COLUMNS = [
        "vehicle_speed",
        "mass",
        "distance",
        "relative_speed",
        "friction",
        "road_environment",
        "weather_condition"
    ]

    def __init__(
        self,
        model_path: str = "aeb_random_forest_model.pkl",
        partial_risk_threshold: float = 0.35,
        full_risk_threshold: float = 0.60,
        ttc_critical: float = 1.4,
        ttc_warning: float = 2.4,
        hysteresis_hold: bool = True,
        enable_physics_fallback: bool = True,
        brake_force_max: float = 12000.0,
        response_delay: float = 0.20,
        margin_critical: float = 1.0,
        margin_warning: float = 4.0,
        g: float = 9.81
    ):
        self.model_path = model_path
        self.model = joblib.load(model_path)
        self.partial_risk_threshold = partial_risk_threshold
        self.full_risk_threshold = full_risk_threshold
        self.ttc_critical = ttc_critical
        self.ttc_warning = ttc_warning
        self.hysteresis_hold = hysteresis_hold
        
        # Physics-based safety fallback parameters
        self.enable_physics_fallback = enable_physics_fallback
        self.brake_force_max = float(brake_force_max)
        self.response_delay = float(response_delay)
        self.margin_critical = float(margin_critical)
        self.margin_warning = float(margin_warning)
        self.g = float(g)

        self.latched_stage: BrakeStage = BrakeStage.OFF

    def compute_ttc(self, distance_m: float, relative_speed_kmh: float) -> float:
        """
        Calculates Time-To-Collision in seconds.
        TTC = distance / relative_speed_ms (if closing, else float('inf')).
        """
        v_rel_ms = relative_speed_kmh / 3.6
        if v_rel_ms > 0.05:  # Closing in on obstacle
            return max(0.0, distance_m / v_rel_ms)
        return float("inf")

    def calculate_stopping_distances(
        self,
        mass_kg: float,
        friction: float,
        relative_speed_kmh: float
    ) -> Tuple[float, float]:
        """
        Calculates critical and warning stopping distances based on physical vehicle mass,
        actuator force capacity, tire-road friction saturation, and actuator latency:
          a_max = min(brake_force_max / mass, friction * g)
          d_stop_full = (v_rel * latency) + (v_rel^2) / (2 * a_max)
          d_crit = d_stop_full + margin_critical
        """
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
        weather_condition: str = "Clear"
    ) -> AEBDecision:
        """
        Evaluates the current simulation state, runs ML model inference,
        checks the kinematic safety stopping distance fallback,
        and returns a structured AEBDecision.
        """
        # If already stopped or obstacle is moving away
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

        # Build feature DataFrame matching trained pipeline schema
        feature_df = pd.DataFrame([{
            "vehicle_speed": float(vehicle_speed_kmh),
            "mass": float(mass_kg),
            "distance": max(0.1, float(distance_m)),
            "relative_speed": max(0.0, float(relative_speed_kmh)),
            "friction": float(friction),
            "road_environment": str(road_environment),
            "weather_condition": str(weather_condition)
        }], columns=self.FEATURE_COLUMNS)

        # Run pipeline inference
        prob_classes = self.model.predict_proba(feature_df)[0]
        # Classes: 0 = No collision risk, 1 = Collision risk
        prob_collision = float(prob_classes[1]) if len(prob_classes) > 1 else float(prob_classes[0])
        raw_pred = int(self.model.predict(feature_df)[0])

        # 1. Base ML control arbitration logic
        stage = BrakeStage.OFF
        reason = "Normal driving"

        if v_rel_ms <= 0.05:
            # Obstacle moving away or speed matched: never trigger emergency braking
            stage = BrakeStage.OFF
            reason = "Obstacle moving away or speed matched"
        else:
            if prob_collision >= self.full_risk_threshold or (ttc <= self.ttc_critical and prob_collision >= 0.20):
                stage = BrakeStage.FULL
                reason = f"Full emergency braking: P(collision)={prob_collision:.2f}, TTC={ttc:.2f}s"
            elif prob_collision >= self.partial_risk_threshold or (ttc <= self.ttc_warning and prob_collision >= 0.15):
                stage = BrakeStage.PARTIAL
                reason = f"Partial warning braking: P(collision)={prob_collision:.2f}, TTC={ttc:.2f}s"

            # 2. Physics-based safety fallback
            # Guarantees that a low ML collision probability cannot suppress a physical collision hazard
            if self.enable_physics_fallback:
                d_crit, d_warn = self.calculate_stopping_distances(mass_kg, friction, relative_speed_kmh)
                if distance_m <= d_crit or ttc <= self.ttc_critical:
                    if stage != BrakeStage.FULL:
                        stage = BrakeStage.FULL
                        reason = f"Physics safety fallback (FULL): gap={distance_m:.1f}m <= d_crit={d_crit:.1f}m, TTC={ttc:.2f}s (ML P={prob_collision:.2f})"
                elif distance_m <= d_warn or ttc <= self.ttc_warning:
                    if stage == BrakeStage.OFF:
                        stage = BrakeStage.PARTIAL
                        reason = f"Physics safety fallback (PARTIAL): gap={distance_m:.1f}m <= d_warn={d_warn:.1f}m, TTC={ttc:.2f}s (ML P={prob_collision:.2f})"

        # 3. Apply hysteresis latching if enabled
        if self.hysteresis_hold:
            # If already at FULL, maintain FULL until vehicle comes to a complete halt or closing speed is eliminated
            if self.latched_stage == BrakeStage.FULL and vehicle_speed_kmh > 0.1 and relative_speed_kmh > 0.0:
                stage = BrakeStage.FULL
                reason += " (Held by AEB hysteresis)"
            elif self.latched_stage == BrakeStage.PARTIAL and stage == BrakeStage.OFF and ttc <= self.ttc_warning:
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
        """Resets the agent's internal latching state."""
        self.latched_stage = BrakeStage.OFF

