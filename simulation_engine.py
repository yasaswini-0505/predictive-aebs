from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

from braking_controller import BrakingController, BrakeStage
from aeb_agent import AEBAgent, AEBDecision


class SimulationOutcome(Enum):
    COLLISION = "COLLISION"
    SAFE_STOP = "SAFE_STOP"
    SAFE_TRANSIT = "SAFE_TRANSIT"
    TIMEOUT = "TIMEOUT"


@dataclass
class TelemetryPoint:
    time: float
    vehicle_speed_kmh: float
    obstacle_speed_kmh: float
    gap_m: float
    relative_speed_kmh: float
    deceleration_ms2: float
    ttc_s: float
    brake_stage: str
    collision_probability: float


@dataclass
class SimulationResult:
    outcome: SimulationOutcome
    collision_occurred: bool
    initial_gap_m: float
    final_gap_m: float
    initial_speed_kmh: float
    final_speed_kmh: float
    impact_speed_kmh: float
    impact_relative_speed_kmh: float
    stopping_distance_m: float
    duration_s: float
    aeb_enabled: bool
    telemetry: List[TelemetryPoint] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "collision": self.collision_occurred,
            "impact_speed_kmh": round(self.impact_speed_kmh, 2),
            "final_gap_m": round(self.final_gap_m, 2),
            "stopping_distance_m": round(self.stopping_distance_m, 2),
            "duration_s": round(self.duration_s, 2),
            "aeb_enabled": self.aeb_enabled
        }


class SimulationEngine:
    """
    Time-stepped closed-loop kinematic simulation environment for AEB systems.
    
    Simulates ego-vehicle approach towards stationary or dynamic lead obstacles,
    supporting both AEB-active and AEB-disabled (baseline human inattention) scenarios.
    """

    def __init__(
        self,
        time_step: float = 0.01,
        max_duration: float = 10.0,
        brake_force_max: float = 12000.0,
        aeb_response_time: float = 0.2,
        agent: Optional[AEBAgent] = None
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
        aeb_enabled: bool = True
    ) -> SimulationResult:
        """
        Executes a single simulation run.
        
        Args:
            initial_speed_kmh: Initial vehicle speed in km/h.
            mass_kg: Vehicle mass in kg.
            initial_distance_m: Initial gap to obstacle in meters.
            relative_speed_kmh: Closing speed (v_ego - v_obs) in km/h.
            friction: Tire-road friction coefficient.
            road_environment: 'Dry', 'Wet', or 'Icy'.
            weather_condition: 'Clear', 'Rain', or 'Fog'.
            aeb_enabled: If True, engages the ML AEBAgent and BrakingController.
                         If False, models unassisted vehicle (driver inattention).
        """
        if initial_speed_kmh < 0:
            raise ValueError("Initial vehicle speed cannot be negative.")
        if initial_distance_m <= 0:
            raise ValueError("Initial distance must be positive.")
        if mass_kg <= 0:
            raise ValueError("Vehicle mass must be positive.")
        if friction <= 0:
            raise ValueError("Friction coefficient must be positive.")

        # Convert initial speeds
        v_ego = initial_speed_kmh / 3.6  # m/s
        v_rel = relative_speed_kmh / 3.6  # m/s
        v_obs = max(v_ego - v_rel, 0.0)  # Obstacle speed in m/s

        gap = float(initial_distance_m)
        current_time = 0.0
        distance_traveled = 0.0

        # Initialize controller
        controller = BrakingController(
            mass=mass_kg,
            friction=friction,
            max_brake_force=self.brake_force_max,
            response_delay=self.aeb_response_time
        )

        # Initialize or reuse agent
        if aeb_enabled:
            agent = self._shared_agent or AEBAgent()
            agent.reset()
        else:
            agent = None

        telemetry: List[TelemetryPoint] = []
        outcome = SimulationOutcome.TIMEOUT
        collision_occurred = False
        impact_speed_kmh = 0.0
        impact_relative_speed_kmh = 0.0

        while current_time < self.max_duration:
            v_ego_kmh = v_ego * 3.6
            v_obs_kmh = v_obs * 3.6
            current_v_rel_kmh = (v_ego - v_obs) * 3.6

            # AEB control decision
            if aeb_enabled and agent is not None:
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
            else:
                deceleration = 0.0
                brake_stage_str = "OFF"
                coll_prob = 0.0
                ttc = (gap / (v_ego - v_obs)) if (v_ego - v_obs) > 0.05 else float("inf")

            # Record telemetry
            telemetry.append(TelemetryPoint(
                time=round(current_time, 3),
                vehicle_speed_kmh=round(v_ego_kmh, 2),
                obstacle_speed_kmh=round(v_obs_kmh, 2),
                gap_m=round(gap, 3),
                relative_speed_kmh=round(current_v_rel_kmh, 2),
                deceleration_ms2=round(deceleration, 3),
                ttc_s=round(ttc, 2) if ttc != float("inf") else 999.0,
                brake_stage=brake_stage_str,
                collision_probability=round(coll_prob, 3)
            ))

            # Integrate kinematics across time step dt
            dt = self.time_step
            # Update ego velocity
            delta_v = deceleration * dt
            if v_ego <= delta_v:
                # Coming to a complete stop within this step
                time_to_stop = v_ego / deceleration if deceleration > 0 else dt
                step_dist = 0.5 * v_ego * time_to_stop
                v_ego = 0.0
            else:
                step_dist = (v_ego - 0.5 * delta_v) * dt
                v_ego -= delta_v

            distance_traveled += step_dist
            closing_step = step_dist - (v_obs * dt)
            gap -= closing_step
            current_time += dt

            # Terminal condition 1: Collision
            if gap <= 0.0:
                outcome = SimulationOutcome.COLLISION
                collision_occurred = True
                impact_speed_kmh = v_ego * 3.6
                impact_relative_speed_kmh = max(0.0, (v_ego - v_obs) * 3.6)
                gap = 0.0
                break

            # Terminal condition 2: Safe stop
            # Vehicle stopped (v_ego <= 0.05 m/s) or matched lead vehicle speed with safe gap
            if v_ego <= 0.05:
                v_ego = 0.0
                outcome = SimulationOutcome.SAFE_STOP
                break

            if v_ego <= v_obs:
                outcome = SimulationOutcome.SAFE_STOP
                break
        else:
            # If loop finished max_duration without collision, scenario transit is safe
            if not collision_occurred:
                outcome = SimulationOutcome.SAFE_TRANSIT

        return SimulationResult(
            outcome=outcome,
            collision_occurred=collision_occurred,
            initial_gap_m=initial_distance_m,
            final_gap_m=max(0.0, gap),
            initial_speed_kmh=initial_speed_kmh,
            final_speed_kmh=v_ego * 3.6,
            impact_speed_kmh=impact_speed_kmh,
            impact_relative_speed_kmh=impact_relative_speed_kmh,
            stopping_distance_m=distance_traveled,
            duration_s=current_time,
            aeb_enabled=aeb_enabled,
            telemetry=telemetry
        )
