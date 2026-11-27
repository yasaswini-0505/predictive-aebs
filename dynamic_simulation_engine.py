"""
Dynamic Lead-Vehicle Simulation Engine for Autonomous Emergency Braking (AEB).

Extends baseline kinematic simulation to support dynamic lead-vehicle behavior:
- Constant speed cruising
- Configurable lead-vehicle deceleration (mild, sudden, emergency braking)
- Configurable lead-vehicle acceleration
- Delayed braking onset (reaction time / cut-in)
- Target velocity clamping (e.g. slowing to a speed or full stop)
- Sub-step numerical integration for precise collision and safe-stop boundary detection
- Detailed telemetry recording including minimum gap and activation latency

Preserves existing production modules by functioning as an advanced experimental extension.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

from braking_controller import BrakingController, BrakeStage
from aeb_agent import AEBAgent, AEBDecision
from simulation_engine import SimulationOutcome


@dataclass
class DynamicTelemetryPoint:
    time: float
    vehicle_speed_kmh: float
    obstacle_speed_kmh: float
    gap_m: float
    relative_speed_kmh: float
    deceleration_ms2: float
    lead_acceleration_ms2: float
    ttc_s: float
    brake_stage: str
    collision_probability: float


@dataclass
class DynamicSimulationResult:
    outcome: SimulationOutcome
    collision_occurred: bool
    initial_gap_m: float
    final_gap_m: float
    min_gap_m: float
    initial_speed_kmh: float
    final_speed_kmh: float
    lead_initial_speed_kmh: float
    lead_final_speed_kmh: float
    impact_speed_kmh: float
    impact_relative_speed_kmh: float
    stopping_distance_m: float
    duration_s: float
    aeb_enabled: bool
    first_brake_time_s: Optional[float] = None
    unnecessary_braking: bool = False
    telemetry: List[DynamicTelemetryPoint] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "collision": self.collision_occurred,
            "impact_speed_kmh": round(self.impact_speed_kmh, 2),
            "impact_relative_speed_kmh": round(self.impact_relative_speed_kmh, 2),
            "final_gap_m": round(self.final_gap_m, 2),
            "min_gap_m": round(self.min_gap_m, 2),
            "first_brake_time_s": round(self.first_brake_time_s, 3) if self.first_brake_time_s is not None else None,
            "stopping_distance_m": round(self.stopping_distance_m, 2),
            "duration_s": round(self.duration_s, 2),
            "aeb_enabled": self.aeb_enabled
        }


class DynamicSimulationEngine:
    """
    Closed-loop kinematic simulation engine supporting dynamic lead-vehicle maneuvers.
    """

    def __init__(
        self,
        time_step: float = 0.01,
        max_duration: float = 10.0,
        brake_force_max: float = 12000.0,
        aeb_response_time: float = 0.20,
        agent: Optional[Any] = None
    ):
        self.time_step = float(time_step)
        self.max_duration = float(max_duration)
        self.brake_force_max = float(brake_force_max)
        self.aeb_response_time = float(aeb_response_time)
        self._shared_agent = agent

    def run_scenario(
        self,
        initial_speed_kmh: float,
        mass_kg: float,
        initial_distance_m: float,
        relative_speed_kmh: float,
        friction: float,
        road_environment: str = "Dry",
        weather_condition: str = "Clear",
        aeb_enabled: bool = True,
        lead_acceleration_ms2: float = 0.0,
        lead_brake_start_time: float = 0.0,
        lead_target_speed_kmh: float = 0.0,
        record_telemetry: bool = True
    ) -> DynamicSimulationResult:
        """
        Executes a simulation run with dynamic lead vehicle kinematics.

        Args:
            initial_speed_kmh: Initial ego vehicle speed (km/h).
            mass_kg: Vehicle mass (kg).
            initial_distance_m: Initial headway gap (m).
            relative_speed_kmh: Initial closing speed (v_ego - v_obs) (km/h).
            friction: Road friction coefficient (mu).
            road_environment: 'Dry', 'Wet', or 'Icy'.
            weather_condition: 'Clear', 'Rain', or 'Fog'.
            aeb_enabled: If True, engages active AEB agent; if False, baseline inattention.
            lead_acceleration_ms2: Lead vehicle acceleration in m/s^2 (< 0 for braking).
            lead_brake_start_time: Timestamp in seconds when lead vehicle begins accelerating/braking.
            lead_target_speed_kmh: Target speed in km/h to clamp lead vehicle at (default 0.0 for full stop).
        """
        if initial_speed_kmh < 0:
            raise ValueError("Initial vehicle speed cannot be negative.")
        if initial_distance_m <= 0:
            raise ValueError("Initial distance must be positive.")
        if mass_kg <= 0:
            raise ValueError("Vehicle mass must be positive.")
        if friction <= 0:
            raise ValueError("Friction coefficient must be positive.")

        # Kinematic conversion to SI units (m/s)
        v_ego = initial_speed_kmh / 3.6
        v_rel_init = relative_speed_kmh / 3.6
        v_obs = max(0.0, v_ego - v_rel_init)
        v_obs_target = max(0.0, lead_target_speed_kmh / 3.6)

        gap = float(initial_distance_m)
        min_gap = gap
        current_time = 0.0
        distance_traveled = 0.0

        controller = BrakingController(
            mass=mass_kg,
            friction=friction,
            max_brake_force=self.brake_force_max,
            response_delay=self.aeb_response_time
        )

        if aeb_enabled:
            agent = self._shared_agent or AEBAgent()
            agent.reset()
        else:
            agent = None

        telemetry: List[DynamicTelemetryPoint] = []
        outcome = SimulationOutcome.TIMEOUT
        collision_occurred = False
        impact_speed_kmh = 0.0
        impact_relative_speed_kmh = 0.0
        first_brake_time_s: Optional[float] = None
        dt = self.time_step

        while current_time < self.max_duration:
            v_ego_kmh = v_ego * 3.6
            v_obs_kmh = v_obs * 3.6
            current_v_rel = v_ego - v_obs
            current_v_rel_kmh = current_v_rel * 3.6

            # Determine dynamic lead vehicle acceleration command for this step
            if current_time >= lead_brake_start_time:
                if lead_acceleration_ms2 < 0.0:
                    # Braking towards target speed
                    if v_obs > v_obs_target:
                        a_lead_step = lead_acceleration_ms2
                    else:
                        a_lead_step = 0.0
                elif lead_acceleration_ms2 > 0.0:
                    # Accelerating towards target speed
                    if v_obs < v_obs_target:
                        a_lead_step = lead_acceleration_ms2
                    else:
                        a_lead_step = 0.0
                else:
                    a_lead_step = 0.0
            else:
                a_lead_step = 0.0

            # AEB controller decision
            if aeb_enabled and agent is not None:
                if getattr(agent, "is_predictive", False):
                    # Optical brake light perception: active when lead is decelerating
                    brake_light = 1 if (a_lead_step < -0.8 and v_obs > 0.1) else 0
                    decision = agent.evaluate_state(
                        vehicle_speed_kmh=v_ego_kmh,
                        mass_kg=mass_kg,
                        distance_m=gap,
                        relative_speed_kmh=current_v_rel_kmh,
                        friction=friction,
                        road_environment=road_environment,
                        weather_condition=weather_condition,
                        lead_deceleration_ms2=a_lead_step,
                        brake_light_detected=brake_light
                    )
                else:
                    decision = agent.evaluate_state(
                        vehicle_speed_kmh=v_ego_kmh,
                        mass_kg=mass_kg,
                        distance_m=gap,
                        relative_speed_kmh=current_v_rel_kmh,
                        friction=friction,
                        road_environment=road_environment,
                        weather_condition=weather_condition
                    )
                controller.set_target_stage(decision.stage, current_time=current_time)
                deceleration = controller.compute_deceleration(current_time, v_ego)
                brake_stage_str = decision.stage.name
                coll_prob = decision.collision_probability
                ttc = decision.ttc

                if decision.stage in (BrakeStage.PARTIAL, BrakeStage.FULL) and first_brake_time_s is None:
                    first_brake_time_s = current_time
            else:
                deceleration = 0.0
                brake_stage_str = "OFF"
                coll_prob = 0.0
                ttc = (gap / current_v_rel) if current_v_rel > 0.05 else float("inf")

            if record_telemetry:
                telemetry.append(DynamicTelemetryPoint(
                    time=round(current_time, 3),
                    vehicle_speed_kmh=round(v_ego_kmh, 2),
                    obstacle_speed_kmh=round(v_obs_kmh, 2),
                    gap_m=round(gap, 3),
                    relative_speed_kmh=round(current_v_rel_kmh, 2),
                    deceleration_ms2=round(deceleration, 3),
                    lead_acceleration_ms2=round(a_lead_step, 3),
                    ttc_s=round(ttc, 2) if ttc != float("inf") else 999.0,
                    brake_stage=brake_stage_str,
                    collision_probability=round(coll_prob, 3)
                ))

            # 1. Integrate ego vehicle kinematics
            delta_v_ego = deceleration * dt
            if v_ego <= delta_v_ego:
                # Sub-step zero speed clamp
                t_stop_ego = v_ego / deceleration if deceleration > 0 else dt
                step_dist_ego = 0.5 * v_ego * t_stop_ego
                v_ego = 0.0
            else:
                step_dist_ego = (v_ego - 0.5 * delta_v_ego) * dt
                v_ego -= delta_v_ego

            distance_traveled += step_dist_ego

            # 2. Integrate lead vehicle kinematics
            delta_v_lead = a_lead_step * dt
            if a_lead_step < 0.0:
                # Decelerating
                if (v_obs + delta_v_lead) <= v_obs_target:
                    # Lead reaches target velocity within this time step
                    if abs(a_lead_step) > 1e-6:
                        t_to_target = max(0.0, (v_obs - v_obs_target) / abs(a_lead_step))
                    else:
                        t_to_target = dt
                    t_to_target = min(dt, t_to_target)
                    step_dist_lead = 0.5 * (v_obs + v_obs_target) * t_to_target + v_obs_target * (dt - t_to_target)
                    v_obs = v_obs_target
                else:
                    step_dist_lead = (v_obs + 0.5 * delta_v_lead) * dt
                    v_obs += delta_v_lead
            elif a_lead_step > 0.0:
                # Accelerating
                if (v_obs + delta_v_lead) >= v_obs_target:
                    if abs(a_lead_step) > 1e-6:
                        t_to_target = max(0.0, (v_obs_target - v_obs) / a_lead_step)
                    else:
                        t_to_target = dt
                    t_to_target = min(dt, t_to_target)
                    step_dist_lead = 0.5 * (v_obs + v_obs_target) * t_to_target + v_obs_target * (dt - t_to_target)
                    v_obs = v_obs_target
                else:
                    step_dist_lead = (v_obs + 0.5 * delta_v_lead) * dt
                    v_obs += delta_v_lead
            else:
                step_dist_lead = v_obs * dt

            # Guarantee non-negative speeds
            v_ego = max(0.0, v_ego)
            v_obs = max(0.0, v_obs)

            # 3. Update spatial gap
            closing_step = step_dist_ego - step_dist_lead
            gap -= closing_step
            min_gap = min(min_gap, gap)
            current_time += dt

            # Terminal condition 1: Collision occurred
            if gap <= 0.0:
                outcome = SimulationOutcome.COLLISION
                collision_occurred = True
                impact_speed_kmh = v_ego * 3.6
                impact_relative_speed_kmh = max(0.0, (v_ego - v_obs) * 3.6)
                gap = 0.0
                min_gap = 0.0
                break

            # Terminal condition 2: Both vehicles safely stopped
            if v_ego <= 0.05 and v_obs <= 0.05:
                v_ego = 0.0
                v_obs = 0.0
                outcome = SimulationOutcome.SAFE_STOP
                break

            # Terminal condition 3: Ego vehicle stopped while lead vehicle is moving away or stopped
            if v_ego <= 0.05 and v_obs >= 0.0:
                v_ego = 0.0
                outcome = SimulationOutcome.SAFE_STOP
                break

            # Terminal condition 4: Ego matched lead speed and lead is not decelerating (stable safe headway)
            if v_ego <= v_obs and a_lead_step >= 0.0 and current_time > (lead_brake_start_time + 0.5):
                outcome = SimulationOutcome.SAFE_STOP
                break
        else:
            if not collision_occurred:
                outcome = SimulationOutcome.SAFE_TRANSIT

        return DynamicSimulationResult(
            outcome=outcome,
            collision_occurred=collision_occurred,
            initial_gap_m=initial_distance_m,
            final_gap_m=max(0.0, gap),
            min_gap_m=max(0.0, min_gap),
            initial_speed_kmh=initial_speed_kmh,
            final_speed_kmh=v_ego * 3.6,
            lead_initial_speed_kmh=v_obs_kmh,
            lead_final_speed_kmh=v_obs * 3.6,
            impact_speed_kmh=impact_speed_kmh,
            impact_relative_speed_kmh=impact_relative_speed_kmh,
            stopping_distance_m=distance_traveled,
            duration_s=current_time,
            aeb_enabled=aeb_enabled,
            first_brake_time_s=first_brake_time_s,
            telemetry=telemetry
        )

