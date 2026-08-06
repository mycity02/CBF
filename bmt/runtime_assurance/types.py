from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class CBFConfig:
    """Configuration for the two-dimensional runtime safety filter.

    The physical limits are deliberately expressed in SI units. The MetaDrive
    adapter is responsible for translating normalized policy actions to and
    from these units.
    """

    dt: float = 0.1
    alpha1: float = 1.25
    alpha2: float = 1.25
    longitudinal_margin: float = 0.75
    lateral_margin: float = 0.25
    road_margin: float = 0.20
    road_activation_h: float = 0.50
    road_activation_psi1: float = 0.0
    sensor_range: float = 60.0
    activation_horizon: float = 3.0
    vulnerable_road_user_activation_horizon: float = 1.5
    activation_steps: int = 20
    activation_h_threshold: float = 0.0
    activation_longitudinal_buffer: float = 0.75
    activation_lateral_buffer: float = 0.25
    activation_use_nominal_control: bool = False
    activation_lateral_gate_horizon: float = 0.0
    passthrough_without_active_obstacle: bool = False
    road_low_speed_passthrough: float = 2.0
    road_low_speed_min_outward_heading_component: float = 0.03
    road_use_side_detector: bool = False
    include_vulnerable_road_users: bool = False
    enforce_road_barriers: bool = True
    infeasible_nominal_preview_fallback: bool = False
    activation_require_current_lateral_overlap: bool = True
    barrier_regularization: float = 1e-6
    other_accel_uncertainty: float = 1.0
    min_acceleration: float = -6.0
    max_acceleration: float = 3.0
    max_curvature: float = 0.30
    acceleration_weight: float = 1.0
    curvature_weight: float = 40.0
    escape_curvature_fraction: float = 0.45
    preview_horizon: float = 1.5
    preview_steps: int = 15
    preview_max_obstacles: int = 0
    preview_safety_weight: float = 8.0
    preview_action_weight: float = 1.0
    preview_safe_saturation: float = 1.0
    fsm_enter_h: float = 0.80
    fsm_exit_h: float = 1.20
    fsm_enter_psi1: float = 0.50
    fsm_exit_psi1: float = 1.00
    fsm_min_dwell_steps: int = 5
    fsm_clear_steps: int = 3
    fsm_recovery_steps: int = 5
    fsm_lateral_target: float = 3.0
    fsm_escape_heading_target: float = 0.18
    fsm_escape_heading_gain: float = 1.5
    fsm_min_side_margin: float = 0.0
    fsm_front_escape_enabled: bool = False
    fsm_front_escape_min_margin: float = 0.20
    fsm_direction_hysteresis: float = 0.05
    fsm_brake_acceleration: float = -3.0
    fsm_brake_ramp_start_steps: int = 10
    fsm_brake_ramp_steps: int = 20
    fsm_recovery_heading_gain: float = 1.0
    fallback_penalty: float = 1000.0
    fallback_iterations: int = 120
    constraint_tolerance: float = 1e-7
    intervention_tolerance: float = 1e-3
    estimated_accel_limit: float = 12.0

    def __post_init__(self):
        positive = {
            "dt": self.dt,
            "alpha1": self.alpha1,
            "alpha2": self.alpha2,
            "sensor_range": self.sensor_range,
            "activation_horizon": self.activation_horizon,
            "vulnerable_road_user_activation_horizon": (
                self.vulnerable_road_user_activation_horizon
            ),
            "activation_steps": self.activation_steps,
            "barrier_regularization": self.barrier_regularization,
            "max_curvature": self.max_curvature,
            "preview_horizon": self.preview_horizon,
            "preview_steps": self.preview_steps,
            "fsm_clear_steps": self.fsm_clear_steps,
            "fsm_recovery_steps": self.fsm_recovery_steps,
            "fsm_lateral_target": self.fsm_lateral_target,
            "fsm_escape_heading_target": self.fsm_escape_heading_target,
            "fsm_escape_heading_gain": self.fsm_escape_heading_gain,
            "fsm_brake_ramp_steps": self.fsm_brake_ramp_steps,
            "fallback_penalty": self.fallback_penalty,
            "fallback_iterations": self.fallback_iterations,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.min_acceleration >= self.max_acceleration:
            raise ValueError("min_acceleration must be less than max_acceleration")
        nonnegative = {
            "activation_h_threshold": self.activation_h_threshold,
            "activation_longitudinal_buffer": self.activation_longitudinal_buffer,
            "activation_lateral_buffer": self.activation_lateral_buffer,
            "road_activation_h": self.road_activation_h,
            "road_activation_psi1": self.road_activation_psi1,
            "activation_lateral_gate_horizon": self.activation_lateral_gate_horizon,
            "road_low_speed_passthrough": self.road_low_speed_passthrough,
            "road_low_speed_min_outward_heading_component": (
                self.road_low_speed_min_outward_heading_component
            ),
            "preview_max_obstacles": self.preview_max_obstacles,
            "fsm_min_dwell_steps": self.fsm_min_dwell_steps,
            "fsm_brake_ramp_start_steps": self.fsm_brake_ramp_start_steps,
            "fsm_min_side_margin": self.fsm_min_side_margin,
            "fsm_front_escape_min_margin": self.fsm_front_escape_min_margin,
            "fsm_direction_hysteresis": self.fsm_direction_hysteresis,
            "fsm_recovery_heading_gain": self.fsm_recovery_heading_gain,
        }
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative, got {value}")
        if self.road_low_speed_min_outward_heading_component >= 1.0:
            raise ValueError(
                "road_low_speed_min_outward_heading_component must be less than 1"
            )
        if not 0.0 < self.escape_curvature_fraction <= 1.0:
            raise ValueError("escape_curvature_fraction must be in (0, 1]")
        if self.fsm_exit_h < self.fsm_enter_h:
            raise ValueError("fsm_exit_h must be greater than or equal to fsm_enter_h")
        if self.fsm_exit_psi1 < self.fsm_enter_psi1:
            raise ValueError(
                "fsm_exit_psi1 must be greater than or equal to fsm_enter_psi1"
            )
        if not self.min_acceleration <= self.fsm_brake_acceleration <= 0.0:
            raise ValueError(
                "fsm_brake_acceleration must be nonpositive and lie inside the acceleration bounds"
            )

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None) -> "CBFConfig":
        return cls(**dict(values or {}))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Control2D:
    acceleration: float
    curvature: float

    def as_array(self) -> np.ndarray:
        return np.asarray([self.acceleration, self.curvature], dtype=float)

    @classmethod
    def from_array(cls, value: np.ndarray) -> "Control2D":
        value = np.asarray(value, dtype=float).reshape(2)
        return cls(acceleration=float(value[0]), curvature=float(value[1]))


@dataclass(frozen=True)
class VehicleState2D:
    name: str
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    heading: float
    length: float
    width: float
    kind: str = "vehicle"

    def __post_init__(self):
        object.__setattr__(self, "position", _vector2(self.position, "position"))
        object.__setattr__(self, "velocity", _vector2(self.velocity, "velocity"))
        object.__setattr__(self, "acceleration", _vector2(self.acceleration, "acceleration"))
        object.__setattr__(self, "kind", str(self.kind).lower())
        if self.length <= 0 or self.width <= 0:
            raise ValueError("vehicle length and width must be positive")

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def heading_vector(self) -> np.ndarray:
        return np.asarray([np.cos(self.heading), np.sin(self.heading)], dtype=float)


@dataclass(frozen=True)
class RoadGeometry:
    """Local road geometry at the ego position.

    Clearances are measured from the ego center to the outer drivable
    boundaries. A missing clearance disables that boundary constraint.
    """

    heading: float
    left_clearance: Optional[float]
    right_clearance: Optional[float]

    @property
    def tangent(self) -> np.ndarray:
        return np.asarray([np.cos(self.heading), np.sin(self.heading)], dtype=float)

    @property
    def left_normal(self) -> np.ndarray:
        return np.asarray([-np.sin(self.heading), np.cos(self.heading)], dtype=float)


@dataclass(frozen=True)
class BarrierDiagnostic:
    name: str
    kind: str
    h: float
    h_dot: float
    psi1: float
    residual: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FilterResult:
    control: Control2D
    nominal_control: Control2D
    mode: str
    intervened: bool
    hard_feasible: bool
    max_constraint_violation: float
    preview_score: float
    normalized_constraint_violation: float = 0.0
    fallback_mode: str = "none"
    mandatory_feasible: bool = True
    active_obstacle_count: int = 0
    diagnostics: List[BarrierDiagnostic] = field(default_factory=list)
    candidate_scores: Dict[str, float] = field(default_factory=dict)
    reference_control: Optional[Control2D] = None
    decision_margin: Optional[float] = None
    previous_mode: Optional[str] = None
    transition_reason: str = "none"
    state_age_steps: int = 0
    fsm_transitioned: bool = False
    fsm_left_margin: Optional[float] = None
    fsm_right_margin: Optional[float] = None
    qp_solve_count: int = 0

    def to_telemetry(self) -> Dict[str, Any]:
        vehicle_h = [item.h for item in self.diagnostics if item.kind == "vehicle"]
        road_h = [item.h for item in self.diagnostics if item.kind == "road"]
        reference = self.reference_control or self.nominal_control
        return {
            "rta_enabled": True,
            "rta_mode": self.mode,
            "rta_intervened": self.intervened,
            "rta_hard_feasible": self.hard_feasible,
            "rta_max_constraint_violation": self.max_constraint_violation,
            "rta_normalized_constraint_violation": self.normalized_constraint_violation,
            "rta_fallback_mode": self.fallback_mode,
            "rta_mandatory_feasible": self.mandatory_feasible,
            "rta_active_obstacle_count": self.active_obstacle_count,
            "rta_preview_score": self.preview_score,
            "rta_decision_margin": self.decision_margin,
            "rta_reference_acceleration": reference.acceleration,
            "rta_reference_curvature": reference.curvature,
            "rta_nominal_acceleration": self.nominal_control.acceleration,
            "rta_nominal_curvature": self.nominal_control.curvature,
            "rta_safe_acceleration": self.control.acceleration,
            "rta_safe_curvature": self.control.curvature,
            "rta_min_vehicle_h": min(vehicle_h) if vehicle_h else None,
            "rta_min_road_h": min(road_h) if road_h else None,
            "rta_candidate_scores": dict(self.candidate_scores),
            "rta_fsm_previous_mode": self.previous_mode,
            "rta_fsm_transition_reason": self.transition_reason,
            "rta_fsm_state_age_steps": self.state_age_steps,
            "rta_fsm_transitioned": self.fsm_transitioned,
            "rta_fsm_left_margin": self.fsm_left_margin,
            "rta_fsm_right_margin": self.fsm_right_margin,
            "rta_qp_solve_count": self.qp_solve_count,
            "rta_barriers": [item.to_dict() for item in self.diagnostics],
        }


def _vector2(value: np.ndarray, name: str) -> np.ndarray:
    ret = np.asarray(value, dtype=float).reshape(-1)
    if ret.size < 2:
        raise ValueError(f"{name} must contain at least two values")
    ret = ret[:2].copy()
    if not np.all(np.isfinite(ret)):
        raise ValueError(f"{name} contains non-finite values")
    return ret
