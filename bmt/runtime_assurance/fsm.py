"""Finite-state supervisory controller for the two-dimensional HOCBF filter.

The supervisor replaces the legacy per-step nominal/left/right candidate
rollouts.  It keeps one persistent behavioral state and solves one HOCBF-QP
for that state's reference control at each control step.
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import List, Sequence, Tuple

import numpy as np

from bmt.runtime_assurance.cbf2d import (
    TwoDimensionalCBF,
    _Barrier,
    _projected_half_extents,
)
from bmt.runtime_assurance.qp import solve_prioritized_projection
from bmt.runtime_assurance.types import (
    BarrierDiagnostic,
    CBFConfig,
    Control2D,
    FilterResult,
    RoadGeometry,
    VehicleState2D,
)


class FSMState(str, Enum):
    NOMINAL = "nominal"
    BRAKE = "brake"
    ESCAPE_LEFT = "escape_left"
    ESCAPE_RIGHT = "escape_right"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class ThreatAssessment:
    front_threat: bool
    rear_threat: bool
    front_clear: bool
    rear_clear: bool
    left_margin: float
    right_margin: float
    front_h: float = float("inf")
    front_psi1: float = float("inf")

    @property
    def risks_clear(self) -> bool:
        return self.front_clear and self.rear_clear


@dataclass(frozen=True)
class FSMDecision:
    state: FSMState
    previous_state: FSMState
    transitioned: bool
    reason: str
    state_age_steps: int
    left_margin: float
    right_margin: float


class SafetySupervisorFSM:
    """Persistent normal/brake/escape/recovery supervisor with hysteresis."""

    def __init__(self, config: CBFConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.state = FSMState.NOMINAL
        self.state_age_steps = 0
        self.clear_steps = 0
        self.last_transition_reason = "reset"

    def update(
        self,
        assessment: ThreatAssessment,
        nominal_curvature: float,
    ) -> FSMDecision:
        previous = self.state
        next_state = previous
        reason = "hold"

        if assessment.risks_clear:
            self.clear_steps += 1
        else:
            self.clear_steps = 0

        if previous is FSMState.NOMINAL:
            if assessment.rear_threat:
                next_state, reason = self._select_escape(
                    assessment,
                    nominal_curvature,
                    prefix="rear_threat",
                )
            elif assessment.front_threat:
                next_state, reason = self._select_front_response(
                    assessment,
                    nominal_curvature,
                    prefix="front_threat",
                )

        elif previous is FSMState.BRAKE:
            # A newly detected rear threat is a safety escalation and is not
            # delayed by the dwell timer.
            if assessment.rear_threat:
                next_state, reason = self._select_escape(
                    assessment,
                    nominal_curvature,
                    prefix="rear_threat_while_braking",
                )
            elif (
                assessment.front_threat
                and self.state_age_steps >= self.config.fsm_min_dwell_steps
            ):
                candidate, candidate_reason = self._select_front_response(
                    assessment,
                    nominal_curvature,
                    prefix="front_threat_while_braking",
                )
                if candidate is not FSMState.BRAKE:
                    next_state = candidate
                    reason = candidate_reason
            elif (
                self.state_age_steps >= self.config.fsm_min_dwell_steps
                and self.clear_steps >= self.config.fsm_clear_steps
            ):
                next_state = FSMState.RECOVERY
                reason = "threat_cleared"

        elif previous in (FSMState.ESCAPE_LEFT, FSMState.ESCAPE_RIGHT):
            selected_margin = (
                assessment.left_margin
                if previous is FSMState.ESCAPE_LEFT
                else assessment.right_margin
            )
            if selected_margin < self.config.fsm_min_side_margin:
                if assessment.rear_threat or assessment.front_threat:
                    next_state, reason = self._select_escape(
                        assessment,
                        nominal_curvature,
                        prefix="selected_side_blocked",
                    )
                else:
                    next_state = FSMState.BRAKE
                    reason = "selected_side_blocked"
            elif (
                self.state_age_steps >= self.config.fsm_min_dwell_steps
                and self.clear_steps >= self.config.fsm_clear_steps
            ):
                next_state = FSMState.RECOVERY
                reason = "threat_cleared"

        elif previous is FSMState.RECOVERY:
            if assessment.rear_threat:
                next_state, reason = self._select_escape(
                    assessment,
                    nominal_curvature,
                    prefix="rear_threat_during_recovery",
                )
            elif assessment.front_threat:
                next_state, reason = self._select_front_response(
                    assessment,
                    nominal_curvature,
                    prefix="front_threat_during_recovery",
                )
            elif (
                assessment.risks_clear
                and self.state_age_steps >= self.config.fsm_recovery_steps
            ):
                next_state = FSMState.NOMINAL
                reason = "recovery_complete"

        transitioned = next_state is not previous
        if transitioned:
            self.state = next_state
            self.state_age_steps = 1
            self.clear_steps = 0
            self.last_transition_reason = reason
        else:
            self.state_age_steps += 1

        return FSMDecision(
            state=self.state,
            previous_state=previous,
            transitioned=transitioned,
            reason=reason,
            state_age_steps=self.state_age_steps,
            left_margin=assessment.left_margin,
            right_margin=assessment.right_margin,
        )

    def _select_escape(
        self,
        assessment: ThreatAssessment,
        nominal_curvature: float,
        prefix: str,
        minimum_margin: float = None,
    ) -> Tuple[FSMState, str]:
        threshold = (
            self.config.fsm_min_side_margin
            if minimum_margin is None
            else max(self.config.fsm_min_side_margin, minimum_margin)
        )
        left_ok = assessment.left_margin >= threshold
        right_ok = assessment.right_margin >= threshold
        if not left_ok and not right_ok:
            return FSMState.BRAKE, f"{prefix}_no_escape_corridor"
        if left_ok and not right_ok:
            return FSMState.ESCAPE_LEFT, f"{prefix}_left_only"
        if right_ok and not left_ok:
            return FSMState.ESCAPE_RIGHT, f"{prefix}_right_only"

        difference = assessment.left_margin - assessment.right_margin
        hysteresis = self.config.fsm_direction_hysteresis
        if difference > hysteresis:
            return FSMState.ESCAPE_LEFT, f"{prefix}_left_margin"
        if difference < -hysteresis:
            return FSMState.ESCAPE_RIGHT, f"{prefix}_right_margin"
        if nominal_curvature < -self.config.intervention_tolerance:
            return FSMState.ESCAPE_RIGHT, f"{prefix}_nominal_preference"
        return FSMState.ESCAPE_LEFT, f"{prefix}_deterministic_tie"

    def _select_front_response(
        self,
        assessment: ThreatAssessment,
        nominal_curvature: float,
        prefix: str,
    ) -> Tuple[FSMState, str]:
        """Choose a route-aware front response configured for this experiment."""

        if not self.config.fsm_front_escape_enabled:
            return FSMState.BRAKE, f"{prefix}_adaptive_brake"
        return self._select_escape(
            assessment,
            nominal_curvature,
            prefix=prefix,
            minimum_margin=self.config.fsm_front_escape_min_margin,
        )


class FSMHOCBF(TwoDimensionalCBF):
    """FSM-supervised, single-QP two-dimensional HOCBF safety filter."""

    def __init__(self, config: CBFConfig = None):
        super().__init__(config)
        self.supervisor = SafetySupervisorFSM(self.config)

    def reset(self) -> None:
        self.supervisor.reset()

    def filter(
        self,
        nominal_control: Control2D,
        ego: VehicleState2D,
        obstacles: Sequence[VehicleState2D],
        road: RoadGeometry,
    ) -> FilterResult:
        sensor_obstacles = [
            obstacle
            for obstacle in obstacles
            if np.linalg.norm(ego.position - obstacle.position)
            <= self.config.sensor_range
        ]
        active_obstacles = self._current_active_obstacles(
            ego,
            sensor_obstacles,
            road,
        )
        assessment = self._assess_threats(
            ego,
            active_obstacles,
            sensor_obstacles,
            road,
        )
        decision = self.supervisor.update(
            assessment,
            nominal_curvature=nominal_control.curvature,
        )

        # Road safety is independent of traffic-threat activation. Evaluate
        # the instantaneous road HOCBF constraints on every control step so a
        # nominal/no-obstacle pass-through cannot silently bypass them.
        all_barriers = self._build_barriers(
            ego=ego,
            obstacles=active_obstacles,
            road=road,
        )
        road_barriers = [item for item in all_barriers if item.diagnostic.kind == "road"]
        road_guard_active = any(
            item.diagnostic.h <= self.config.road_activation_h
            or item.diagnostic.psi1 <= self.config.road_activation_psi1
            for item in road_barriers
        )
        barriers = [
            item
            for item in all_barriers
            if item.diagnostic.kind != "road" or road_guard_active
        ]
        nominal_array = nominal_control.as_array()
        road_nominal_safe = all(
            item.constraint.residual(nominal_array) >= -self.config.constraint_tolerance
            for item in road_barriers
        )
        passthrough_diagnostics = self._executed_diagnostics(
            road_barriers,
            nominal_control,
        )

        low_speed_road_recovery = (
            self.config.road_low_speed_passthrough > 0.0
            and not active_obstacles
            and ego.speed < self.config.road_low_speed_passthrough
            and self._inside_physical_road_but_outside_margin(ego, road)
        )
        inactive_passthrough = (
            self.config.passthrough_without_active_obstacle
            and not active_obstacles
            and decision.state is FSMState.NOMINAL
            and (not road_guard_active or road_nominal_safe)
        )
        if inactive_passthrough or low_speed_road_recovery:
            return FilterResult(
                control=nominal_control,
                nominal_control=nominal_control,
                mode=decision.state.value,
                intervened=False,
                hard_feasible=True,
                max_constraint_violation=0.0,
                preview_score=self.config.preview_safe_saturation,
                normalized_constraint_violation=0.0,
                fallback_mode=(
                    "low_speed_road_recovery" if low_speed_road_recovery else "inactive"
                ),
                mandatory_feasible=True,
                active_obstacle_count=0,
                diagnostics=passthrough_diagnostics,
                reference_control=nominal_control,
                decision_margin=self._selected_margin(decision),
                previous_mode=decision.previous_state.value,
                transition_reason=decision.reason,
                state_age_steps=decision.state_age_steps,
                fsm_transitioned=decision.transitioned,
                fsm_left_margin=decision.left_margin,
                fsm_right_margin=decision.right_margin,
                qp_solve_count=0,
            )

        constraints = [item.constraint for item in barriers]
        reference = self._reference_for_state(
            decision.state,
            nominal_control,
            ego,
            road,
            assessment,
        )
        solution = solve_prioritized_projection(
            reference=reference.as_array(),
            weights=np.asarray(
                [self.config.acceleration_weight, self.config.curvature_weight],
                dtype=float,
            ),
            constraints=constraints,
            lower_bounds=np.asarray(
                [self.config.min_acceleration, -self.config.max_curvature],
                dtype=float,
            ),
            upper_bounds=np.asarray(
                [self.config.max_acceleration, self.config.max_curvature],
                dtype=float,
            ),
            tolerance=self.config.constraint_tolerance,
            fallback_penalty=self.config.fallback_penalty,
            fallback_iterations=self.config.fallback_iterations,
        )
        safe_control = Control2D.from_array(solution.control)
        diagnostics = self._executed_diagnostics(all_barriers, safe_control)
        executed_violations = [
            max(0.0, -item.constraint.residual(safe_control.as_array()))
            for item in barriers
        ]
        difference = safe_control.as_array() - nominal_control.as_array()
        decision_margin = self._selected_margin(decision)

        return FilterResult(
            control=safe_control,
            nominal_control=nominal_control,
            mode=decision.state.value,
            intervened=bool(
                np.linalg.norm(difference) > self.config.intervention_tolerance
            ),
            hard_feasible=solution.hard_feasible,
            max_constraint_violation=max(executed_violations, default=0.0),
            preview_score=(
                decision_margin
                if decision_margin is not None
                else self.config.preview_safe_saturation
            ),
            normalized_constraint_violation=solution.normalized_violation,
            fallback_mode=solution.fallback_mode,
            mandatory_feasible=solution.mandatory_feasible,
            active_obstacle_count=len(active_obstacles),
            diagnostics=diagnostics,
            reference_control=reference,
            decision_margin=decision_margin,
            previous_mode=decision.previous_state.value,
            transition_reason=decision.reason,
            state_age_steps=decision.state_age_steps,
            fsm_transitioned=decision.transitioned,
            fsm_left_margin=decision.left_margin,
            fsm_right_margin=decision.right_margin,
            qp_solve_count=1,
        )

    def _current_active_obstacles(
        self,
        ego: VehicleState2D,
        obstacles: Sequence[VehicleState2D],
        road: RoadGeometry,
    ) -> List[VehicleState2D]:
        active: List[VehicleState2D] = []
        for obstacle in obstacles:
            diagnostic = self._vehicle_barrier(ego, obstacle, road).diagnostic
            if (
                diagnostic.h <= self.config.fsm_exit_h
                or diagnostic.psi1 <= self.config.fsm_exit_psi1
            ):
                active.append(obstacle)
        return active

    def _assess_threats(
        self,
        ego: VehicleState2D,
        active_obstacles: Sequence[VehicleState2D],
        sensor_obstacles: Sequence[VehicleState2D],
        road: RoadGeometry,
    ) -> ThreatAssessment:
        front_threat = False
        rear_threat = False
        front_clear = True
        rear_clear = True
        front_h = float("inf")
        front_psi1 = float("inf")
        tangent = road.tangent
        for obstacle in active_obstacles:
            diagnostic = self._vehicle_barrier(ego, obstacle, road).diagnostic
            longitudinal = float((obstacle.position - ego.position) @ tangent)
            # A rear vehicle can become a genuine 2-D sideswipe/rear threat
            # before its footprint fully overlaps the ego laterally. Keep the
            # overlap gate for front traffic, but let the current HOCBF state
            # trigger an early rear escape without a trajectory prediction.
            if (
                longitudinal >= 0.0
                and not self._current_lateral_overlap(ego, obstacle, road)
            ):
                continue
            entering = (
                diagnostic.h <= self.config.fsm_enter_h
                or diagnostic.psi1 <= self.config.fsm_enter_psi1
            )
            clear = (
                diagnostic.h >= self.config.fsm_exit_h
                and diagnostic.psi1 >= self.config.fsm_exit_psi1
            )
            if longitudinal >= 0.0:
                front_threat = front_threat or entering
                front_clear = front_clear and clear
                front_h = min(front_h, diagnostic.h)
                front_psi1 = min(front_psi1, diagnostic.psi1)
            else:
                rear_threat = rear_threat or entering
                rear_clear = rear_clear and clear

        left_margin = self._direction_margin(
            1.0,
            ego,
            sensor_obstacles,
            road,
        )
        right_margin = self._direction_margin(
            -1.0,
            ego,
            sensor_obstacles,
            road,
        )
        return ThreatAssessment(
            front_threat=front_threat,
            rear_threat=rear_threat,
            front_clear=front_clear,
            rear_clear=rear_clear,
            left_margin=left_margin,
            right_margin=right_margin,
            front_h=front_h,
            front_psi1=front_psi1,
        )

    def _direction_margin(
        self,
        direction: float,
        ego: VehicleState2D,
        obstacles: Sequence[VehicleState2D],
        road: RoadGeometry,
    ) -> float:
        clearance = road.left_clearance if direction > 0.0 else road.right_clearance
        lateral_extent = _projected_half_extents(
            ego,
            road.tangent,
            road.left_normal,
        )[1]
        if clearance is None or not np.isfinite(clearance):
            target = self.config.fsm_lateral_target
            road_score = 1.0
        else:
            available = float(clearance) - lateral_extent - self.config.road_margin
            if available <= 0.0:
                return float(available / max(ego.width, 1.0))
            target = min(self.config.fsm_lateral_target, 0.70 * available)
            road_score = (available - target) / max(ego.width, 1.0)

        if target <= 0.0:
            return float("-inf")
        virtual_ego = replace(
            ego,
            position=ego.position + direction * target * road.left_normal,
        )
        scores: List[float] = [float(road_score)]
        for obstacle in obstacles:
            diagnostic = self._vehicle_barrier(
                virtual_ego,
                obstacle,
                road,
            ).diagnostic
            scores.append(diagnostic.h)
            original_lateral = float(
                (obstacle.position - ego.position) @ road.left_normal
            )
            if direction * original_lateral > 0.5 * ego.width:
                scores.append(diagnostic.psi1 / max(self.config.alpha1, 1e-6))
        return float(min(scores))

    def _reference_for_state(
        self,
        state: FSMState,
        nominal: Control2D,
        ego: VehicleState2D,
        road: RoadGeometry,
        assessment: ThreatAssessment = None,
    ) -> Control2D:
        if state is FSMState.BRAKE:
            return Control2D(
                acceleration=self._adaptive_brake_acceleration(
                    nominal.acceleration,
                    assessment,
                ),
                curvature=nominal.curvature,
            )
        if state in (FSMState.ESCAPE_LEFT, FSMState.ESCAPE_RIGHT):
            direction = 1.0 if state is FSMState.ESCAPE_LEFT else -1.0
            curvature = self._escape_feedback_curvature(
                direction,
                ego,
                road,
            )
            return Control2D(
                acceleration=nominal.acceleration,
                curvature=float(
                    np.clip(
                        curvature,
                        -self.config.max_curvature,
                        self.config.max_curvature,
                    )
                ),
            )
        if state is FSMState.RECOVERY:
            heading_error = _wrap_angle(ego.heading - road.heading)
            correction = (
                -self.config.fsm_recovery_heading_gain
                * heading_error
                / max(ego.speed, 1.0)
            )
            return Control2D(
                acceleration=nominal.acceleration,
                curvature=float(
                    np.clip(
                        nominal.curvature + correction,
                        -self.config.max_curvature,
                        self.config.max_curvature,
                    )
                ),
            )
        return nominal

    def _adaptive_brake_acceleration(
        self,
        nominal_acceleration: float,
        assessment: ThreatAssessment,
    ) -> float:
        """Scale braking continuously with the instantaneous front risk.

        The HOCBF-QP still enforces the hard vehicle constraint. This method
        only changes its reference: a threshold-level threat requests coasting,
        while h <= 0 or psi1 <= 0 requests the configured full brake. Once the
        entering condition clears, the nominal acceleration is released even
        while FSM hysteresis keeps the controller in BRAKE.
        """

        if assessment is None:
            return float(
                min(nominal_acceleration, self.config.fsm_brake_acceleration)
            )
        if not assessment.front_threat:
            return float(nominal_acceleration)

        h_scale = max(abs(self.config.fsm_enter_h), 1e-6)
        psi1_scale = max(abs(self.config.fsm_enter_psi1), 1e-6)
        h_risk = np.clip(
            (self.config.fsm_enter_h - assessment.front_h) / h_scale,
            0.0,
            1.0,
        )
        psi1_risk = np.clip(
            (self.config.fsm_enter_psi1 - assessment.front_psi1) / psi1_scale,
            0.0,
            1.0,
        )
        raw_risk = float(max(h_risk, psi1_risk))
        persistent_risk = np.clip(
            (
                self.supervisor.state_age_steps
                - self.config.fsm_brake_ramp_start_steps
            )
            / max(self.config.fsm_brake_ramp_steps, 1),
            0.0,
            1.0,
        )
        risk = float(max(raw_risk, persistent_risk))
        target = risk * self.config.fsm_brake_acceleration
        return float(min(nominal_acceleration, target))

    def _escape_feedback_curvature(
        self,
        direction: float,
        ego: VehicleState2D,
        road: RoadGeometry,
    ) -> float:
        """Track an escape heading using only the current measured state.

        This is an algebraic feedback law, not a trajectory rollout or a
        finite-horizon motion prediction. The available current road
        clearance attenuates the target heading as the boundary approaches.
        """

        clearance = road.left_clearance if direction > 0.0 else road.right_clearance
        if clearance is None or not np.isfinite(clearance):
            clearance_scale = 1.0
        else:
            available = max(
                0.0,
                float(clearance)
                - _projected_half_extents(
                    ego,
                    road.tangent,
                    road.left_normal,
                )[1]
                - self.config.road_margin,
            )
            clearance_scale = float(
                np.clip(
                    available / max(self.config.fsm_lateral_target, 1e-6),
                    0.0,
                    1.0,
                )
            )
        heading_error = _wrap_angle(ego.heading - road.heading)
        target_heading_error = (
            direction
            * self.config.fsm_escape_heading_target
            * clearance_scale
        )
        curvature = self.config.fsm_escape_heading_gain * (
            target_heading_error - heading_error
        ) / max(ego.speed, 1.0)
        limit = (
            self.config.max_curvature
            * self.config.escape_curvature_fraction
        )
        return float(np.clip(curvature, -limit, limit))

    @staticmethod
    def _executed_diagnostics(
        barriers: Sequence[_Barrier],
        control: Control2D,
    ) -> List[BarrierDiagnostic]:
        return [
            BarrierDiagnostic(
                name=item.diagnostic.name,
                kind=item.diagnostic.kind,
                h=item.diagnostic.h,
                h_dot=item.diagnostic.h_dot,
                psi1=item.diagnostic.psi1,
                residual=item.constraint.residual(control.as_array()),
            )
            for item in barriers
        ]

    @staticmethod
    def _selected_margin(decision: FSMDecision):
        if decision.state is FSMState.ESCAPE_LEFT:
            return decision.left_margin
        if decision.state is FSMState.ESCAPE_RIGHT:
            return decision.right_margin
        if decision.state is FSMState.BRAKE:
            return min(decision.left_margin, decision.right_margin)
        return None


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
