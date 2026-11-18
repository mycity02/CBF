"""Finite-state supervisory controller for the two-dimensional HOCBF filter.

The supervisor replaces the legacy per-step nominal/left/right candidate
rollouts.  It keeps one persistent behavioral state and solves one HOCBF-QP
for that state's reference control at each control step.
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np

from cbf_rta.cbf2d import (
    TwoDimensionalCBF,
    _Barrier,
    _projected_half_extents,
)
from cbf_rta.qp import (
    LinearConstraint,
    solve_prioritized_projection,
    solve_weighted_projection,
)
from cbf_rta.types import (
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


@dataclass(frozen=True)
class _RoadRecoveryDecision:
    """One-step command for a latched low-speed road recovery."""

    reference: Optional[Control2D] = None
    side: Optional[str] = None
    use_envelope: bool = False
    fallback_mode: Optional[str] = None


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
            if (self.config.rear_release_completed_escape and assessment.rear_clear
                    and not assessment.rear_threat
                    and self.state_age_steps >= self.config.fsm_min_dwell_steps):
                next_state = FSMState.BRAKE if assessment.front_threat else FSMState.RECOVERY
                reason = ('rear_clear_front_brake' if assessment.front_threat else
                          'rear_clear_physical_recovery')
            elif selected_margin < self.config.fsm_min_side_margin:
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
        if (
            self.config.road_low_speed_passthrough
            > self.config.fsm_corner_road_cbf_max_speed
        ):
            raise ValueError(
                "road_low_speed_passthrough must not exceed "
                "fsm_corner_road_cbf_max_speed for FSMHOCBF"
            )
        self.supervisor = SafetySupervisorFSM(self.config)
        self._road_recovery_barrier_name = None

    def reset(self) -> None:
        self.supervisor.reset()
        self._road_recovery_barrier_name = None

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
            nominal_control,
        )
        decision = self.supervisor.update(
            assessment,
            nominal_curvature=nominal_control.curvature,
        )
        self._prepare_maneuver(ego, sensor_obstacles, road, decision)

        # Road safety is independent of traffic-threat activation. Evaluate
        # the instantaneous road HOCBF constraints on every control step so a
        # nominal/no-obstacle pass-through cannot silently bypass them.
        all_barriers = self._build_barriers(
            ego=ego,
            obstacles=active_obstacles,
            road=road,
        )
        road_barriers = [
            item for item in all_barriers if item.diagnostic.kind == "road"
        ]
        vehicle_barriers = [
            item for item in all_barriers if item.diagnostic.kind == "vehicle"
        ]
        road_guard_active = any(
            item.diagnostic.h <= self.config.road_activation_h
            or item.diagnostic.psi1 <= self.config.road_activation_psi1
            for item in road_barriers
        )
        if self.config.rear_use_native_dynamics:
            # The physical profile must not let a vehicle-risk correction
            # bypass road constraints simply because the current clearance is large.
            road_guard_active = True
        road_recovery = self._road_recovery_reference(
            nominal_control,
            ego,
            road_barriers,
            vehicle_barriers=vehicle_barriers,
        )
        barriers: List[_Barrier] = []
        diagnostic_barriers: List[_Barrier] = []
        for item in all_barriers:
            effective_item = item
            if item.diagnostic.kind == "road" and not road_guard_active:
                diagnostic_barriers.append(item)
                continue
            if (
                road_recovery.use_envelope
                and road_recovery.side is not None
                and item.diagnostic.name.startswith(road_recovery.side)
            ):
                effective_item = self._recovery_envelope_barrier(item)
            barriers.append(effective_item)
            diagnostic_barriers.append(effective_item)
        nominal_array = nominal_control.as_array()
        road_nominal_safe = all(
            item.constraint.residual(nominal_array) >= -self.config.constraint_tolerance
            for item in road_barriers
        )
        passthrough_diagnostics = self._executed_diagnostics(
            road_barriers,
            nominal_control,
        )

        inactive_passthrough = (
            self.config.passthrough_without_active_obstacle
            and self._allow_inactive_passthrough()
            and not active_obstacles
            and decision.state is FSMState.NOMINAL
            and (not road_guard_active or road_nominal_safe)
            and road_recovery.reference is None
            and (not self.config.rear_use_native_dynamics or (
                self.config.min_acceleration <= nominal_control.acceleration <= self.config.max_acceleration
                and abs(nominal_control.curvature) <= self.config.max_curvature
            ))
        )
        if inactive_passthrough:
            return FilterResult(
                control=nominal_control,
                nominal_control=nominal_control,
                mode=decision.state.value,
                intervened=False,
                hard_feasible=True,
                max_constraint_violation=0.0,
                preview_score=self.config.preview_safe_saturation,
                normalized_constraint_violation=0.0,
                fallback_mode="inactive",
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

        if road_recovery.fallback_mode == "road_recovery_fail_stop":
            fail_stop = road_recovery.reference
            diagnostics = self._executed_diagnostics(
                all_barriers,
                fail_stop,
            )
            active_constraints = [item.constraint for item in barriers]
            violations = [
                max(0.0, -item.residual(fail_stop.as_array()))
                for item in active_constraints
            ]
            difference = fail_stop.as_array() - nominal_control.as_array()
            return FilterResult(
                control=fail_stop,
                nominal_control=nominal_control,
                mode=decision.state.value,
                intervened=bool(
                    np.linalg.norm(difference) > self.config.intervention_tolerance
                ),
                hard_feasible=False,
                max_constraint_violation=max(violations, default=0.0),
                preview_score=self.config.preview_safe_saturation,
                normalized_constraint_violation=(
                    self._normalized_constraint_violation(
                        active_constraints,
                        fail_stop,
                    )
                ),
                fallback_mode=road_recovery.fallback_mode,
                mandatory_feasible=False,
                active_obstacle_count=len(active_obstacles),
                diagnostics=diagnostics,
                reference_control=fail_stop,
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
        reference = (
            road_recovery.reference
            if road_recovery.reference is not None
            else self._reference_for_state(
                decision.state,
                nominal_control,
                ego,
                road,
                assessment,
            )
        )
        recovery_fallback_mode = road_recovery.fallback_mode
        vehicle_priority = recovery_fallback_mode == "road_recovery_conflict"
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
            mandatory_kinds=("vehicle",) if vehicle_priority else ("road",),
            tolerance=self.config.constraint_tolerance,
            fallback_penalty=self.config.fallback_penalty,
            fallback_iterations=self.config.fallback_iterations,
            precondition=self.config.qp_preconditioned_fallback,
        )
        if road_recovery.use_envelope and not solution.hard_feasible:
            vehicle_priority = True
            recovery_fallback_mode = "road_recovery_conflict"
            solution = solve_prioritized_projection(
                reference=reference.as_array(),
                weights=np.asarray(
                    [
                        self.config.acceleration_weight,
                        self.config.curvature_weight,
                    ],
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
                mandatory_kinds=("vehicle",),
                tolerance=self.config.constraint_tolerance,
                fallback_penalty=self.config.fallback_penalty,
                fallback_iterations=self.config.fallback_iterations,
                precondition=self.config.qp_preconditioned_fallback,
            )
        safe_control = Control2D.from_array(solution.control)
        diagnostics = self._executed_diagnostics(
            diagnostic_barriers,
            safe_control,
        )
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
            fallback_mode=self._combined_fallback_mode(
                recovery_fallback_mode,
                self._solver_fallback_mode(
                    solution.fallback_mode,
                    vehicle_priority=vehicle_priority,
                ),
            ),
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

    def _prepare_maneuver(self, ego, obstacles, road, decision) -> None:
        """Optional extension point; the legacy filter changes no state here."""

    def _allow_inactive_passthrough(self) -> bool:
        return True

    def _current_active_obstacles(
        self,
        ego: VehicleState2D,
        obstacles: Sequence[VehicleState2D],
        road: RoadGeometry,
    ) -> List[VehicleState2D]:
        active: List[VehicleState2D] = []
        for obstacle in obstacles:
            barrier = self._vehicle_barrier(ego, obstacle, road)
            diagnostic = barrier.diagnostic
            if not (
                diagnostic.h <= self.config.fsm_exit_h
                or diagnostic.psi1 <= self.config.fsm_exit_psi1
            ):
                continue
            if (
                self.config.activation_require_current_lateral_overlap
                and obstacle.kind == "vehicle"
                and not self._current_lateral_overlap(ego, obstacle, road)
                and not self._non_overlapping_vehicle_is_converging(
                    ego,
                    obstacle,
                    road,
                    envelope_h_dot=diagnostic.h_dot,
                )
            ):
                continue
            active.append(obstacle)
        return active

    def _assess_threats(
        self,
        ego: VehicleState2D,
        active_obstacles: Sequence[VehicleState2D],
        sensor_obstacles: Sequence[VehicleState2D],
        road: RoadGeometry,
        nominal_control: Control2D,
    ) -> ThreatAssessment:
        front_threat = False
        rear_threat = False
        front_clear = True
        rear_clear = True
        front_h = float("inf")
        front_psi1 = float("inf")
        tangent = road.tangent
        nominal_array = nominal_control.as_array()
        for obstacle in active_obstacles:
            barrier = self._vehicle_barrier(ego, obstacle, road)
            diagnostic = barrier.diagnostic
            longitudinal = float((obstacle.position - ego.position) @ tangent)
            entering = (
                diagnostic.h <= self.config.fsm_enter_h
                or diagnostic.psi1 <= self.config.fsm_enter_psi1
            )
            clear = (
                diagnostic.h >= self.config.fsm_exit_h
                and diagnostic.psi1 >= self.config.fsm_exit_psi1
            )
            if longitudinal >= 0.0:
                nominal_unsafe = (
                    barrier.constraint.residual(nominal_array)
                    < -self.config.constraint_tolerance
                )
                front_threat = front_threat or (
                    entering and (diagnostic.h <= 0.0 or nominal_unsafe)
                )
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

    def _non_overlapping_vehicle_is_converging(
        self,
        ego: VehicleState2D,
        obstacle: VehicleState2D,
        road: RoadGeometry,
        envelope_h_dot: float,
    ) -> bool:
        """Keep only plausible cut-ins or closing offset rear vehicles."""

        if envelope_h_dot >= -self.config.fsm_barrier_rate_epsilon:
            return False
        closing_speed_epsilon = self.config.fsm_closing_speed_epsilon
        relative_position = obstacle.position - ego.position
        relative_velocity = obstacle.velocity - ego.velocity
        longitudinal = float(relative_position @ road.tangent)
        lateral = float(relative_position @ road.left_normal)
        lateral_rate = float(relative_velocity @ road.left_normal)
        lateral_converging = (
            lateral * lateral_rate
            < -self.config.fsm_lateral_convergence_product_epsilon
        )
        if lateral_converging:
            return True
        if longitudinal >= 0.0:
            return False

        longitudinal_closing = (
            float(relative_velocity @ road.tangent) > closing_speed_epsilon
        )
        _, lateral_extent = self._combined_extents(
            ego,
            obstacle,
            road.tangent,
            road.left_normal,
        )
        return longitudinal_closing and abs(lateral) <= lateral_extent

    def _road_recovery_reference(
        self,
        nominal: Control2D,
        ego: VehicleState2D,
        road_barriers: Sequence[_Barrier],
        vehicle_barriers: Sequence[_Barrier] = (),
    ) -> _RoadRecoveryDecision:
        """Return a directed control that restores a slightly violated road set.

        A relative-degree-two road HOCBF cannot guarantee forward invariance
        when an episode starts outside its safe set, and at near-zero speed its
        curvature coefficient also vanishes. A one-sided violation therefore
        latches an inward reference until the nominal safe set is recovered.
        The bounded recovery envelope is used only while the original corner
        constraints have no intersection with the physical input box. The
        opposite road boundary and all vehicle constraints remain in the QP.
        """

        by_side = {
            side: [
                item for item in road_barriers if item.diagnostic.name.startswith(side)
            ]
            for side in ("road:left", "road:right")
        }
        active_side = self._road_recovery_barrier_name
        if active_side is not None:
            active_barriers = by_side.get(active_side, [])
            opposite_side = "road:right" if active_side == "road:left" else "road:left"
            opposite_violated = any(
                item.diagnostic.h < 0.0 for item in by_side.get(opposite_side, [])
            )
            if (
                not active_barriers
                or opposite_violated
                or min(item.diagnostic.h for item in active_barriers)
                < -self.config.fsm_road_recovery_max_initial_penetration
            ):
                self._road_recovery_barrier_name = None
                if opposite_violated or active_barriers:
                    return self._road_conflict_or_fail_stop(
                        nominal,
                        None if opposite_violated else active_side,
                        vehicle_barriers,
                    )
                return _RoadRecoveryDecision()
            if min(
                item.diagnostic.h for item in active_barriers
            ) >= self.config.fsm_road_recovery_release_h and all(
                item.constraint.residual(nominal.as_array())
                >= -self.config.constraint_tolerance
                for item in active_barriers
            ):
                self._road_recovery_barrier_name = None
                active_side = None

        if active_side is None:
            if (
                self.config.road_low_speed_passthrough <= 0.0
                or ego.speed >= self.config.road_low_speed_passthrough
            ):
                return _RoadRecoveryDecision()
            negative_sides = [
                side
                for side, items in by_side.items()
                if items and min(item.diagnostic.h for item in items) < 0.0
            ]
            if len(negative_sides) > 1:
                return self._road_conflict_or_fail_stop(
                    nominal,
                    None,
                    vehicle_barriers,
                )
            violated_sides = [
                side
                for side, items in by_side.items()
                if items
                and -self.config.fsm_road_recovery_max_initial_penetration
                <= min(item.diagnostic.h for item in items)
                < 0.0
            ]
            if len(violated_sides) != 1:
                if negative_sides:
                    return self._road_conflict_or_fail_stop(
                        nominal,
                        negative_sides[0],
                        vehicle_barriers,
                    )
                return _RoadRecoveryDecision()
            active_side = violated_sides[0]
            self._road_recovery_barrier_name = active_side

        raw_road_feasible = self._road_constraints_are_feasible(road_barriers)
        use_envelope = not raw_road_feasible
        curvature = self._inward_recovery_curvature(active_side, nominal)
        reference = Control2D(
            acceleration=(
                self.config.fsm_road_recovery_acceleration
                if use_envelope
                else float(
                    np.clip(
                        nominal.acceleration,
                        self.config.min_acceleration,
                        self.config.max_acceleration,
                    )
                )
            ),
            curvature=curvature,
        )
        return _RoadRecoveryDecision(
            reference=reference,
            side=active_side,
            use_envelope=use_envelope,
            fallback_mode=(
                "low_speed_road_recovery" if use_envelope else "directed_road_recovery"
            ),
        )

    def _road_constraints_are_feasible(
        self,
        road_barriers: Sequence[_Barrier],
    ) -> bool:
        solution = solve_weighted_projection(
            reference=np.zeros(2, dtype=float),
            weights=np.ones(2, dtype=float),
            constraints=[item.constraint for item in road_barriers],
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
        return solution.hard_feasible

    def _inward_recovery_curvature(
        self,
        active_side: str,
        nominal: Control2D,
    ) -> float:
        magnitude = (
            self.config.max_curvature * self.config.fsm_road_recovery_curvature_fraction
        )
        if active_side == "road:left":
            curvature = min(nominal.curvature, -magnitude)
        else:
            curvature = max(nominal.curvature, magnitude)
        return float(
            np.clip(
                curvature,
                -self.config.max_curvature,
                self.config.max_curvature,
            )
        )

    def _road_fail_stop(
        self,
        nominal: Control2D,
        active_side: Optional[str],
    ) -> _RoadRecoveryDecision:
        self._road_recovery_barrier_name = None
        curvature = (
            0.0
            if active_side is None
            else (
                -self.config.max_curvature
                if active_side == "road:left"
                else self.config.max_curvature
            )
        )
        return _RoadRecoveryDecision(
            reference=Control2D(
                acceleration=self.config.min_acceleration,
                curvature=curvature,
            ),
            side=active_side,
            fallback_mode="road_recovery_fail_stop",
        )

    def _road_conflict_or_fail_stop(
        self,
        nominal: Control2D,
        active_side: Optional[str],
        vehicle_barriers: Sequence[_Barrier],
    ) -> _RoadRecoveryDecision:
        self._road_recovery_barrier_name = None
        fail_stop = self._road_fail_stop(nominal, active_side)
        fail_stop_control = fail_stop.reference.as_array()
        if any(
            item.constraint.residual(fail_stop_control)
            < -self.config.constraint_tolerance
            for item in vehicle_barriers
        ):
            return _RoadRecoveryDecision(
                fallback_mode="road_recovery_conflict",
            )
        return fail_stop

    @staticmethod
    def _combined_fallback_mode(
        recovery_mode: Optional[str],
        solver_mode: str,
    ) -> str:
        if recovery_mode is None:
            return solver_mode
        if solver_mode == "none":
            return recovery_mode
        return f"{recovery_mode}+{solver_mode}"

    @staticmethod
    def _solver_fallback_mode(
        solver_mode: str,
        vehicle_priority: bool,
    ) -> str:
        if not vehicle_priority:
            return solver_mode
        if solver_mode == "relaxed_vehicle":
            return "relaxed_road"
        if solver_mode == "mandatory_infeasible":
            return "vehicle_mandatory_infeasible"
        return solver_mode

    def _normalized_constraint_violation(
        self,
        constraints: Sequence[LinearConstraint],
        control: Control2D,
    ) -> float:
        if not constraints:
            return 0.0
        spans = np.asarray(
            [
                self.config.max_acceleration - self.config.min_acceleration,
                2.0 * self.config.max_curvature,
            ],
            dtype=float,
        )
        values = []
        for item in constraints:
            scale = max(
                1.0,
                abs(item.lower),
                float(np.sum(np.abs(item.coefficients) * spans)),
            )
            values.append(max(0.0, -item.residual(control.as_array())) / scale)
        return max(values, default=0.0)

    def _road_barriers(
        self,
        ego: VehicleState2D,
        road: RoadGeometry,
    ) -> List[_Barrier]:
        """Use corner CBFs only where the road HOCBF loses steering authority.

        The legacy projected-footprint barrier changes with vehicle heading,
        but its derivative freezes that footprint. Corner barriers retain the
        missing heading-rate term at low speed, making curvature an immediate
        road-safety input. At normal speed the second-order road HOCBF is kept:
        it is less sensitive to piecewise changes of the local route tangent.
        Vehicle interactions always remain relative-degree-two HOCBF constraints.
        """

        use_corner_barriers = self._road_recovery_barrier_name is not None or (
            self.config.fsm_corner_road_cbf_max_speed > 0.0
            and ego.speed < self.config.fsm_corner_road_cbf_max_speed
        )
        if not use_corner_barriers:
            return super()._road_barriers(ego, road)

        return self._corner_road_barriers(ego, road)

    def _corner_road_barriers(self, ego, road) -> List[_Barrier]:
        """First-order corner constraints including the footprint yaw term."""

        heading = ego.heading_vector
        left_heading = np.asarray([-heading[1], heading[0]], dtype=float)
        corner_specs = []
        for longitudinal_name, longitudinal_sign in (
            ("front", 1.0),
            ("rear", -1.0),
        ):
            for lateral_name, lateral_sign in (
                ("left", 1.0),
                ("right", -1.0),
            ):
                offset = (
                    longitudinal_sign * 0.5 * ego.length * heading
                    + lateral_sign * 0.5 * ego.width * left_heading
                )
                corner_specs.append((f"{longitudinal_name}_{lateral_name}", offset))

        barriers: List[_Barrier] = []
        boundary_specs = (
            ("road:left", road.left_clearance, -road.left_normal),
            ("road:right", road.right_clearance, road.left_normal),
        )
        for side_name, clearance, safe_gradient in boundary_specs:
            if clearance is None or not np.isfinite(clearance):
                continue
            center_drift = float(safe_gradient @ ego.velocity)
            for corner_name, offset in corner_specs:
                rotated_offset = np.asarray([-offset[1], offset[0]], dtype=float)
                h = float(clearance + safe_gradient @ offset - self.config.road_margin)
                curvature_coefficient = float(
                    ego.speed * (safe_gradient @ rotated_offset)
                )
                lower = -center_drift - self.config.alpha1 * h
                name = f"{side_name}:{corner_name}"
                barriers.append(
                    _Barrier(
                        constraint=LinearConstraint(
                            coefficients=np.asarray(
                                [0.0, curvature_coefficient],
                                dtype=float,
                            ),
                            lower=lower,
                            name=name,
                            kind="road",
                        ),
                        diagnostic=BarrierDiagnostic(
                            name=name,
                            kind="road",
                            h=h,
                            h_dot=center_drift,
                            psi1=float(center_drift + self.config.alpha1 * h),
                        ),
                    )
                )
        return barriers

    def _recovery_envelope_barrier(self, barrier: _Barrier) -> _Barrier:
        """Shift one road side to the configured bounded recovery set."""

        recovery_delta = self.config.fsm_road_recovery_max_initial_penetration
        return _Barrier(
            constraint=LinearConstraint(
                coefficients=barrier.constraint.coefficients.copy(),
                lower=(
                    barrier.constraint.lower
                    - self.config.alpha1 * recovery_delta
                ),
                name=barrier.constraint.name,
                kind=barrier.constraint.kind,
            ),
            diagnostic=replace(
                barrier.diagnostic,
                nominal_h=barrier.diagnostic.h,
                effective_h=barrier.diagnostic.h + recovery_delta,
            ),
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
            return float(min(nominal_acceleration, self.config.fsm_brake_acceleration))
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
            (self.supervisor.state_age_steps - self.config.fsm_brake_ramp_start_steps)
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
            direction * self.config.fsm_escape_heading_target * clearance_scale
        )
        curvature = (
            self.config.fsm_escape_heading_gain
            * (target_heading_error - heading_error)
            / max(ego.speed, 1.0)
        )
        limit = self.config.max_curvature * self.config.escape_curvature_fraction
        return float(np.clip(curvature, -limit, limit))

    def _executed_diagnostics(
        self,
        barriers: Sequence[_Barrier],
        control: Control2D,
    ) -> List[BarrierDiagnostic]:
        diagnostics: List[BarrierDiagnostic] = []
        control_array = control.as_array()
        for item in barriers:
            residual = item.constraint.residual(control_array)
            h_dot = item.diagnostic.h_dot
            psi1 = item.diagnostic.psi1
            nominal_h = (
                item.diagnostic.nominal_h
                if item.diagnostic.nominal_h is not None
                else item.diagnostic.h
            )
            effective_h = (
                item.diagnostic.effective_h
                if item.diagnostic.effective_h is not None
                else item.diagnostic.h
            )
            if item.diagnostic.kind == "road" and item.diagnostic.name.count(":") >= 2:
                h_dot += float(item.constraint.coefficients[1] * control.curvature)
                psi1 = h_dot + self.config.alpha1 * nominal_h
            diagnostics.append(
                BarrierDiagnostic(
                    name=item.diagnostic.name,
                    kind=item.diagnostic.kind,
                    h=item.diagnostic.h,
                    h_dot=h_dot,
                    psi1=psi1,
                    residual=residual,
                    nominal_h=nominal_h,
                    effective_h=effective_h,
                    effective_residual=residual,
                    control_coefficients=item.constraint.coefficients.tolist(),
                    lower_bound=float(item.constraint.lower),
                )
            )
        return diagnostics

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
