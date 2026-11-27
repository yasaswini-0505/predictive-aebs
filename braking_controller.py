from enum import Enum
from typing import Optional


class BrakeStage(Enum):
    """Braking intensity stages for Autonomous Emergency Braking (AEB)."""
    OFF = 0.0          # No AEB braking intervention
    PARTIAL = 0.4      # Pre-braking / warning phase deceleration (40% max effort)
    FULL = 1.0         # Emergency braking intervention (100% max available effort)


class BrakingController:
    """
    Physics-based vehicle braking controller.
    
    Models brake actuator limits, tire-road surface friction saturation,
    multi-stage deceleration demands, and system activation latency.
    
    Attributes:
        mass (float): Vehicle mass in kg (typically 1000 - 2000 kg).
        friction (float): Road surface friction coefficient mu (e.g., 0.8 Dry, 0.5 Wet, 0.2 Icy).
        max_brake_force (float): Maximum braking force capacity in Newtons (default: 12000 N).
        response_delay (float): System activation latency in seconds (default: 0.2 s).
        g (float): Gravitational acceleration (default: 9.81 m/s^2).
    """

    def __init__(
        self,
        mass: float = 1500.0,
        friction: float = 0.8,
        max_brake_force: float = 12000.0,
        response_delay: float = 0.2,
        g: float = 9.81
    ):
        if mass <= 0:
            raise ValueError("Vehicle mass must be positive.")
        if friction <= 0:
            raise ValueError("Friction coefficient must be positive.")
        if max_brake_force <= 0:
            raise ValueError("Max brake force must be positive.")
        if response_delay < 0:
            raise ValueError("Response delay cannot be negative.")

        self.mass = float(mass)
        self.friction = float(friction)
        self.max_brake_force = float(max_brake_force)
        self.response_delay = float(response_delay)
        self.g = float(g)

        self.current_stage = BrakeStage.OFF
        self.target_stage = BrakeStage.OFF
        self.activation_time: Optional[float] = None
        self.is_active = False

    @property
    def max_available_deceleration(self) -> float:
        """
        Calculates maximum physical deceleration limited by both actuator force and tire friction.
        a_max = min(F_brake_max / mass, mu * g)
        """
        force_limited_accel = self.max_brake_force / self.mass
        friction_limited_accel = self.friction * self.g
        return min(force_limited_accel, friction_limited_accel)

    def set_target_stage(self, stage: BrakeStage, current_time: float) -> None:
        """
        Commands a braking stage change. If transitioning from OFF to active,
        records the activation timestamp to enforce response delay.
        """
        if stage != BrakeStage.OFF and self.target_stage == BrakeStage.OFF:
            self.activation_time = current_time
            self.is_active = True
        elif stage == BrakeStage.OFF:
            self.activation_time = None
            self.is_active = False
            self.current_stage = BrakeStage.OFF

        self.target_stage = stage

    def compute_deceleration(self, current_time: float, current_speed_ms: float) -> float:
        """
        Computes the effective braking deceleration in m/s^2 at the given simulation time.
        
        Args:
            current_time (float): Current simulation time in seconds.
            current_speed_ms (float): Current vehicle speed in m/s.
            
        Returns:
            float: Braking deceleration (>= 0 m/s^2).
        """
        if current_speed_ms <= 0 or self.target_stage == BrakeStage.OFF:
            return 0.0

        if self.activation_time is None:
            return 0.0

        elapsed_since_activation = current_time - self.activation_time

        # Latency check: if within actuator delay window, no braking force applied yet
        # Using tolerance for floating-point arithmetic precision
        if elapsed_since_activation < (self.response_delay - 1e-7):
            return 0.0

        # After response delay, apply target braking intensity
        self.current_stage = self.target_stage
        requested_intensity = self.current_stage.value
        deceleration = requested_intensity * self.max_available_deceleration

        return deceleration

    def reset(self) -> None:
        """Resets the controller state to initial conditions."""
        self.current_stage = BrakeStage.OFF
        self.target_stage = BrakeStage.OFF
        self.activation_time = None
        self.is_active = False
