from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from cbf_rta.qp import (
    LinearConstraint,
    QPSolution,
    solve_prioritized_projection,
)
from cbf_rta.types import (
    BarrierDiagnostic,
    CBFConfig,
    Control2D,
    FilterResult,
    RoadGeometry,
    VehicleState2D,
)


@dataclass(frozen=True)
class _Barrier:
    constraint: LinearConstraint
    diagnostic: BarrierDiagnostic


@dataclass(frozen=True)
class _Candidate:
    mode: str
    solution: QPSolution
    score: float
    preview_margin: float


class TwoDimensionalCBF:
    """Two-dimensional HOCBF safety filter with predictive escape candidates.

    Vehicle footprints are conservatively enclosed by a smooth fourth-order
    superellipse in the local road frame. Each barrier is converted to a
    relative-degree-two HOCBF constraint that is affine in longitudinal
    acceleration and path curvature.

    Three references (nominal, left escape, right escape) are projected onto
    the same safe input set. A short constant-control preview selects the
    candidate with the best future safety margin. This makes the safety filter
    capable of breaking the left/right symmetry of a centered rear threat
    without embedding a lane-specific rule in the CBF itself.
    """

    def __init__(self, config: CBFConfig = None):
        self.config = config or CBFConfig()

    def filter(
        self,
        nominal_control: Control2D,
        ego: VehicleState2D,
        obstacles: Sequence[VehicleState2D],
        road: RoadGeometry,
    ) -> FilterResult:
        preview_obstacles = [
            obstacle
            for obstacle in obstacles
            if np.linalg.norm(ego.position - obstacle.position) <= self.config.sensor_range
        ]
        active_obstacles = self._active_obstacles(
            ego,
            preview_obstacles,
            road,
            nominal_control,
        )
        scoring_obstacles = self._scoring_obstacles(
            ego,
            preview_obstacles,
            active_obstacles,
        )
        low_speed_road_recovery = (
            self.config.road_low_speed_passthrough > 0.0
            and not active_obstacles
            and ego.speed < self.config.road_low_speed_passthrough
            and self._inside_physical_road_but_outside_margin(ego, road)
        )
        if (
            self.config.passthrough_without_active_obstacle and not active_obstacles
        ) or low_speed_road_recovery:
            preview_margin = self._preview_min_margin(
                nominal_control,
                ego,
                scoring_obstacles,
                road,
            )
            return FilterResult(
                control=nominal_control,
                nominal_control=nominal_control,
                mode="nominal",
                intervened=False,
                hard_feasible=True,
                max_constraint_violation=0.0,
                normalized_constraint_violation=0.0,
                fallback_mode=(
                    "low_speed_road_recovery"
                    if low_speed_road_recovery
                    else "inactive"
                ),
                mandatory_feasible=True,
                active_obstacle_count=0,
                preview_score=preview_margin,
                diagnostics=[],
                candidate_scores={"nominal": preview_margin},
                qp_solve_count=0,
            )
        barriers = self._build_barriers(ego=ego, obstacles=active_obstacles, road=road)
        constraints = [item.constraint for item in barriers]
        references = self._candidate_references(nominal_control)

        candidates: List[_Candidate] = []
        candidate_scores: Dict[str, float] = {}
        for mode, reference in references:
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
            control = Control2D.from_array(solution.control)
            preview_margin = self._preview_min_margin(
                control,
                ego,
                scoring_obstacles,
                road,
            )
            score = self._candidate_score(
                control=control,
                nominal_control=nominal_control,
                preview_margin=preview_margin,
                solution=solution,
            )
            candidates.append(
                _Candidate(
                    mode=mode,
                    solution=solution,
                    score=score,
                    preview_margin=preview_margin,
                )
            )
            candidate_scores[mode] = score

        # Stable ordering supplies a deterministic tie break. In an exactly
        # symmetric scene either lateral direction is mathematically valid.
        selected = max(candidates, key=lambda item: item.score)
        safe_control = Control2D.from_array(selected.solution.control)
        fallback_mode = selected.solution.fallback_mode
        selected_mode = selected.mode
        preview_margin = selected.preview_margin
        if self.config.infeasible_nominal_preview_fallback and not selected.solution.hard_feasible:
            nominal_preview = self._preview_min_margin(
                nominal_control, ego, scoring_obstacles, road
            )
            if nominal_preview > preview_margin:
                safe_control = nominal_control
                preview_margin = nominal_preview
                fallback_mode = "nominal_preview"
                selected_mode = "nominal"
        diagnostics = [
            BarrierDiagnostic(
                name=item.diagnostic.name,
                kind=item.diagnostic.kind,
                h=item.diagnostic.h,
                h_dot=item.diagnostic.h_dot,
                psi1=item.diagnostic.psi1,
                residual=item.constraint.residual(safe_control.as_array()),
            )
            for item in barriers
        ]

        executed_control = safe_control.as_array()
        executed_violations = [
            max(0.0, -item.constraint.residual(executed_control)) for item in barriers
        ]
        max_constraint_violation = max(executed_violations, default=0.0)
        difference = safe_control.as_array() - nominal_control.as_array()
        intervened = bool(np.linalg.norm(difference) > self.config.intervention_tolerance)
        return FilterResult(
            control=safe_control,
            nominal_control=nominal_control,
            mode=selected_mode,
            intervened=intervened,
            hard_feasible=selected.solution.hard_feasible,
            max_constraint_violation=max_constraint_violation,
            normalized_constraint_violation=selected.solution.normalized_violation,
            fallback_mode=fallback_mode,
            mandatory_feasible=selected.solution.mandatory_feasible,
            active_obstacle_count=len(active_obstacles),
            preview_score=preview_margin,
            diagnostics=diagnostics,
            candidate_scores=candidate_scores,
            qp_solve_count=len(references),
        )


    def _scoring_obstacles(
        self,
        ego: VehicleState2D,
        preview_obstacles: Sequence[VehicleState2D],
        active_obstacles: Sequence[VehicleState2D],
    ) -> List[VehicleState2D]:
        """Keep every active threat plus the nearest preview-only obstacles.

        The cap only limits candidate rollout scoring. It never removes an
        active obstacle from the CBF constraints.
        """
        limit = self.config.preview_max_obstacles
        if limit <= 0 or len(preview_obstacles) <= limit:
            return list(preview_obstacles)

        active_ids = {id(obstacle) for obstacle in active_obstacles}
        selected = list(active_obstacles)
        remaining = sorted(
            (
                obstacle
                for obstacle in preview_obstacles
                if id(obstacle) not in active_ids
                and obstacle.kind == "vehicle"
            ),
            key=lambda obstacle: float(
                np.linalg.norm(ego.position - obstacle.position)
            ),
        )
        budget = max(0, limit - len(selected))
        selected.extend(remaining[:budget])
        return selected
    def _candidate_references(self, nominal: Control2D) -> List[Tuple[str, Control2D]]:
        escape = self.config.max_curvature * self.config.escape_curvature_fraction
        left_curvature = max(escape, nominal.curvature)
        right_curvature = min(-escape, nominal.curvature)
        return [
            ("nominal", nominal),
            ("left_escape", Control2D(nominal.acceleration, left_curvature)),
            ("right_escape", Control2D(nominal.acceleration, right_curvature)),
        ]

    def _candidate_score(
        self,
        control: Control2D,
        nominal_control: Control2D,
        preview_margin: float,
        solution: QPSolution,
    ) -> float:
        saturated_safety = min(preview_margin, self.config.preview_safe_saturation)
        accel_scale = max(
            self.config.max_acceleration - self.config.min_acceleration,
            1e-6,
        )
        curvature_scale = max(2.0 * self.config.max_curvature, 1e-6)
        action_cost = (
            ((control.acceleration - nominal_control.acceleration) / accel_scale) ** 2
            + ((control.curvature - nominal_control.curvature) / curvature_scale) ** 2
        )
        score = (
            self.config.preview_safety_weight * saturated_safety
            - self.config.preview_action_weight * action_cost
        )
        if not solution.hard_feasible:
            score -= self.config.preview_safety_weight * (
                1.0 + solution.normalized_violation
            )
        if not solution.mandatory_feasible:
            score -= 10.0 * self.config.preview_safety_weight
        return float(score)

    def _active_obstacles(
        self,
        ego: VehicleState2D,
        obstacles: Sequence[VehicleState2D],
        road: RoadGeometry,
        nominal_control: Control2D = None,
    ) -> List[VehicleState2D]:
        active: List[VehicleState2D] = []
        for obstacle in obstacles:
            if np.linalg.norm(ego.position - obstacle.position) > self.config.sensor_range:
                continue
            if (
                self.config.activation_require_current_lateral_overlap
                and obstacle.kind == "vehicle"
                and not self._current_lateral_overlap(ego, obstacle, road)
            ):
                gate_horizon = self.config.activation_lateral_gate_horizon
                if (
                    gate_horizon <= 0.0
                    or self._activation_min_h(
                        ego,
                        obstacle,
                        road,
                        nominal_control,
                        horizon=gate_horizon,
                    )
                    > self.config.activation_h_threshold
                ):
                    continue
            if (
                self._activation_min_h(ego, obstacle, road, nominal_control)
                <= self.config.activation_h_threshold
            ):
                active.append(obstacle)
        return active

    def _current_lateral_overlap(
        self,
        ego: VehicleState2D,
        obstacle: VehicleState2D,
        road: RoadGeometry,
    ) -> bool:
        tangent = road.tangent
        normal = road.left_normal
        _, lateral_extent = self._combined_extents(
            ego=ego,
            obstacle=obstacle,
            tangent=tangent,
            normal=normal,
        )
        physical_extent = lateral_extent / (2.0 ** 0.25)
        overlap_extent = physical_extent + self.config.activation_lateral_buffer
        lateral_separation = abs(float((ego.position - obstacle.position) @ normal))
        return lateral_separation <= overlap_extent

    def _inside_physical_road_but_outside_margin(
        self,
        ego: VehicleState2D,
        road: RoadGeometry,
    ) -> bool:
        lateral_extent = _projected_half_extents(
            ego,
            road.tangent,
            road.left_normal,
        )[1]
        boundary_margins = []
        if road.left_clearance is not None and np.isfinite(road.left_clearance):
            boundary_margins.append(
                (float(road.left_clearance) - lateral_extent, road.left_normal)
            )
        if road.right_clearance is not None and np.isfinite(road.right_clearance):
            boundary_margins.append(
                (float(road.right_clearance) - lateral_extent, -road.left_normal)
            )
        if not boundary_margins:
            return False
        minimum_physical_margin, outward_normal = min(
            boundary_margins,
            key=lambda item: item[0],
        )
        minimum_desired_margin = minimum_physical_margin - self.config.road_margin
        outward_heading_component = float(ego.heading_vector @ outward_normal)
        return (
            minimum_physical_margin >= 0.0
            and minimum_desired_margin < 0.0
            and abs(outward_heading_component)
            >= self.config.road_low_speed_min_outward_heading_component
        )

    def _activation_min_h(
        self,
        ego: VehicleState2D,
        obstacle: VehicleState2D,
        road: RoadGeometry,
        nominal_control: Control2D = None,
        horizon: float = None,
    ) -> float:
        tangent = road.tangent
        normal = road.left_normal
        longitudinal_extent, lateral_extent = self._combined_extents(
            ego=ego,
            obstacle=obstacle,
            tangent=tangent,
            normal=normal,
        )
        longitudinal_extent += self.config.activation_longitudinal_buffer
        lateral_extent += self.config.activation_lateral_buffer

        relative_position = ego.position - obstacle.position
        relative_velocity = ego.velocity - obstacle.velocity
        ego_acceleration = np.zeros(2, dtype=float)
        if self.config.activation_use_nominal_control and nominal_control is not None:
            heading = ego.heading_vector
            left_heading = np.asarray([-heading[1], heading[0]], dtype=float)
            ego_acceleration = (
                nominal_control.acceleration * heading
                + ego.speed ** 2 * nominal_control.curvature * left_heading
            )
        relative_acceleration = ego_acceleration - obstacle.acceleration
        minimum = float("inf")
        boundary = (1.0 + self.config.barrier_regularization) ** 0.25
        activation_horizon = (
            self.config.activation_horizon
            if horizon is None
            else min(self.config.activation_horizon, horizon)
        )
        if obstacle.kind != "vehicle":
            activation_horizon = min(
                activation_horizon,
                self.config.vulnerable_road_user_activation_horizon,
            )
        for time in np.linspace(
            0.0,
            activation_horizon,
            self.config.activation_steps + 1,
        ):
            predicted = (
                relative_position
                + relative_velocity * time
                + 0.5 * relative_acceleration * time ** 2
            )
            longitudinal = float(predicted @ tangent)
            lateral = float(predicted @ normal)
            q = (
                (longitudinal / longitudinal_extent) ** 4
                + (lateral / lateral_extent) ** 4
            )
            h = (q + self.config.barrier_regularization) ** 0.25 - boundary
            minimum = min(minimum, float(h))
        return minimum

    def _build_barriers(
        self,
        ego: VehicleState2D,
        obstacles: Sequence[VehicleState2D],
        road: RoadGeometry,
    ) -> List[_Barrier]:
        barriers: List[_Barrier] = []
        for obstacle in obstacles:
            barriers.append(self._vehicle_barrier(ego, obstacle, road))
        if self.config.enforce_road_barriers:
            barriers.extend(self._road_barriers(ego, road))
        return barriers

    def _vehicle_barrier(
        self,
        ego: VehicleState2D,
        obstacle: VehicleState2D,
        road: RoadGeometry,
    ) -> _Barrier:
        h, gradient, hessian = self._vehicle_geometry(ego, obstacle, road)
        relative_velocity = ego.velocity - obstacle.velocity
        h_dot = float(gradient @ relative_velocity)
        drift = float(
            relative_velocity @ hessian @ relative_velocity
            - gradient @ obstacle.acceleration
        )

        heading = ego.heading_vector
        left_heading = np.asarray([-heading[1], heading[0]], dtype=float)
        coefficients = np.asarray(
            [
                gradient @ heading,
                ego.speed ** 2 * (gradient @ left_heading),
            ],
            dtype=float,
        )
        robust_loss = self.config.other_accel_uncertainty * float(np.linalg.norm(gradient))
        lower = (
            -drift
            + robust_loss
            - (self.config.alpha1 + self.config.alpha2) * h_dot
            - self.config.alpha1 * self.config.alpha2 * h
        )
        name = f"{obstacle.kind}:{obstacle.name}"
        return _Barrier(
            constraint=LinearConstraint(coefficients, lower, name, "vehicle"),
            diagnostic=BarrierDiagnostic(
                name=name,
                kind="vehicle",
                h=float(h),
                h_dot=h_dot,
                psi1=float(h_dot + self.config.alpha1 * h),
            ),
        )

    def _vehicle_geometry(
        self,
        ego: VehicleState2D,
        obstacle: VehicleState2D,
        road: RoadGeometry,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        tangent = road.tangent
        normal = road.left_normal
        relative_position = ego.position - obstacle.position
        longitudinal = float(relative_position @ tangent)
        lateral = float(relative_position @ normal)

        longitudinal_extent, lateral_extent = self._combined_extents(
            ego=ego,
            obstacle=obstacle,
            tangent=tangent,
            normal=normal,
        )
        q = (
            (longitudinal / longitudinal_extent) ** 4
            + (lateral / lateral_extent) ** 4
        )
        gradient_q = (
            4.0 * longitudinal ** 3 / longitudinal_extent ** 4 * tangent
            + 4.0 * lateral ** 3 / lateral_extent ** 4 * normal
        )
        hessian_q = (
            12.0 * longitudinal ** 2 / longitudinal_extent ** 4 * np.outer(tangent, tangent)
            + 12.0 * lateral ** 2 / lateral_extent ** 4 * np.outer(normal, normal)
        )
        regularized_q = q + self.config.barrier_regularization
        boundary = (1.0 + self.config.barrier_regularization) ** 0.25
        h = regularized_q ** 0.25 - boundary
        gradient = 0.25 * regularized_q ** -0.75 * gradient_q
        hessian = (
            0.25 * regularized_q ** -0.75 * hessian_q
            - 0.1875 * regularized_q ** -1.75 * np.outer(gradient_q, gradient_q)
        )
        return float(h), gradient, hessian

    def _combined_extents(
        self,
        ego: VehicleState2D,
        obstacle: VehicleState2D,
        tangent: np.ndarray,
        normal: np.ndarray,
    ) -> Tuple[float, float]:
        ego_long, ego_lat = _projected_half_extents(ego, tangent, normal)
        other_long, other_lat = _projected_half_extents(obstacle, tangent, normal)
        # Scaling by 2**(1/4) makes the fourth-order superellipse contain
        # the full axis-aligned Minkowski bounding box of both vehicles.
        enclosing_scale = 2.0 ** 0.25
        longitudinal = enclosing_scale * (
            ego_long + other_long + self.config.longitudinal_margin
        )
        lateral = enclosing_scale * (
            ego_lat + other_lat + self.config.lateral_margin
        )
        return max(longitudinal, 1e-3), max(lateral, 1e-3)

    def _road_barriers(self, ego: VehicleState2D, road: RoadGeometry) -> List[_Barrier]:
        ret: List[_Barrier] = []
        normal = road.left_normal
        lateral_extent = _projected_half_extents(
            ego,
            road.tangent,
            road.left_normal,
        )[1]

        if road.left_clearance is not None and np.isfinite(road.left_clearance):
            ret.append(
                self._linear_position_barrier(
                    name="road:left",
                    h=float(road.left_clearance - lateral_extent - self.config.road_margin),
                    safe_gradient=-normal,
                    ego=ego,
                )
            )
        if road.right_clearance is not None and np.isfinite(road.right_clearance):
            ret.append(
                self._linear_position_barrier(
                    name="road:right",
                    h=float(road.right_clearance - lateral_extent - self.config.road_margin),
                    safe_gradient=normal,
                    ego=ego,
                )
            )
        return ret

    def _linear_position_barrier(
        self,
        name: str,
        h: float,
        safe_gradient: np.ndarray,
        ego: VehicleState2D,
    ) -> _Barrier:
        h_dot = float(safe_gradient @ ego.velocity)
        heading = ego.heading_vector
        left_heading = np.asarray([-heading[1], heading[0]], dtype=float)
        coefficients = np.asarray(
            [
                safe_gradient @ heading,
                ego.speed ** 2 * (safe_gradient @ left_heading),
            ],
            dtype=float,
        )
        lower = (
            -(self.config.alpha1 + self.config.alpha2) * h_dot
            - self.config.alpha1 * self.config.alpha2 * h
        )
        return _Barrier(
            constraint=LinearConstraint(coefficients, lower, name, "road"),
            diagnostic=BarrierDiagnostic(
                name=name,
                kind="road",
                h=h,
                h_dot=h_dot,
                psi1=float(h_dot + self.config.alpha1 * h),
            ),
        )

    def _preview_min_margin(
        self,
        control: Control2D,
        ego: VehicleState2D,
        obstacles: Sequence[VehicleState2D],
        road: RoadGeometry,
    ) -> float:
        dt = self.config.preview_horizon / self.config.preview_steps
        ego_position = ego.position.copy()
        ego_velocity = ego.velocity.copy()
        ego_heading = float(ego.heading)
        ego_speed = ego.speed
        obstacle_states = [
            {
                "base": obstacle,
                "position": obstacle.position.copy(),
                "velocity": obstacle.velocity.copy(),
            }
            for obstacle in obstacles
            if np.linalg.norm(ego.position - obstacle.position) <= self.config.sensor_range
        ]
        initial_position = ego.position.copy()
        margins: List[float] = []

        for _ in range(self.config.preview_steps + 1):
            predicted_ego = VehicleState2D(
                name=ego.name,
                position=ego_position,
                velocity=ego_velocity,
                acceleration=np.zeros(2),
                heading=ego_heading,
                length=ego.length,
                width=ego.width,
            )
            for state in obstacle_states:
                base = state["base"]
                predicted_obstacle = VehicleState2D(
                    name=base.name,
                    position=state["position"],
                    velocity=state["velocity"],
                    acceleration=base.acceleration,
                    heading=base.heading,
                    length=base.length,
                    width=base.width,
                )
                h, _, _ = self._vehicle_geometry(predicted_ego, predicted_obstacle, road)
                margins.append(h)

            lateral_displacement = float((ego_position - initial_position) @ road.left_normal)
            road_scale = max(ego.width, 1.0)
            if road.left_clearance is not None:
                left_h = (
                    road.left_clearance
                    - lateral_displacement
                    - ego.width / 2.0
                    - self.config.road_margin
                )
                margins.append(left_h / road_scale)
            if road.right_clearance is not None:
                right_h = (
                    road.right_clearance
                    + lateral_displacement
                    - ego.width / 2.0
                    - self.config.road_margin
                )
                margins.append(right_h / road_scale)

            left_heading = np.asarray([-np.sin(ego_heading), np.cos(ego_heading)], dtype=float)
            heading = np.asarray([np.cos(ego_heading), np.sin(ego_heading)], dtype=float)
            ego_acceleration = control.acceleration * heading + ego_speed ** 2 * control.curvature * left_heading
            ego_position = ego_position + ego_velocity * dt + 0.5 * ego_acceleration * dt ** 2
            ego_speed = max(0.0, ego_speed + control.acceleration * dt)
            ego_heading = ego_heading + ego_speed * control.curvature * dt
            ego_velocity = ego_speed * np.asarray([np.cos(ego_heading), np.sin(ego_heading)], dtype=float)

            for state in obstacle_states:
                base = state["base"]
                state["position"] = (
                    state["position"] + state["velocity"] * dt + 0.5 * base.acceleration * dt ** 2
                )
                state["velocity"] = state["velocity"] + base.acceleration * dt

        return float(min(margins, default=self.config.preview_safe_saturation))


def _projected_half_extents(
    vehicle: VehicleState2D,
    tangent: np.ndarray,
    normal: np.ndarray,
) -> Tuple[float, float]:
    heading = vehicle.heading_vector
    side = np.asarray([-heading[1], heading[0]], dtype=float)
    longitudinal = (
        vehicle.length / 2.0 * abs(float(heading @ tangent))
        + vehicle.width / 2.0 * abs(float(side @ tangent))
    )
    lateral = (
        vehicle.length / 2.0 * abs(float(heading @ normal))
        + vehicle.width / 2.0 * abs(float(side @ normal))
    )
    return longitudinal, lateral
