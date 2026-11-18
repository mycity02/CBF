"""Maneuver-conditioned rear protection; no rollouts or action candidates.

An opt-in experimental rear constraint is a first-order, velocity-aware CBF,
not a second-order derivative of the legacy position envelope. It is disabled
by default pending comparative validation; the maneuver/target improvements
can be evaluated separately with the original HOCBF. Safety is conditional
on h >= 0, valid disturbance bounds, feasible constraints and model/sampling
assumptions; out-of-set recovery is NOT a safety guarantee.
"""
from dataclasses import replace

import numpy as np

from cbf_rta.cbf2d import _Barrier, _projected_half_extents
from cbf_rta.fsm import FSMHOCBF, FSMState
from cbf_rta.qp import LinearConstraint
from cbf_rta.types import BarrierDiagnostic, Control2D


REAR_MODEL = "rear_velocity_support_cbf"
ROAD_MODEL = "road_velocity_corner_cbf"


def footprint_support(vehicle, normal):
    """Support of a fourth-order enclosing superellipse.

    Returns its support, gradient w.r.t. normal, and heading derivative. Every
    corner of the rectangular footprint belongs to this enclosing shape.
    The dual 4/3-norm has continuous first derivatives, sufficient here because
    this rear CBF is differentiated once, not twice.
    """
    heading = vehicle.heading_vector
    lateral = np.array([-heading[1], heading[0]])
    axes = 2.0 ** 0.25 * np.array([vehicle.length, vehicle.width]) / 2.0
    projections = np.array([normal @ heading, normal @ lateral])
    components = axes * projections
    q = 4.0 / 3.0
    power = max(float(np.sum(np.abs(components) ** q)), 1e-15)
    radius = power ** (1.0 / q)
    gradient = power ** (1.0 / q - 1.0) * np.sign(components) * np.abs(components) ** (q - 1.0)
    grad_normal = gradient[0] * axes[0] * heading + gradient[1] * axes[1] * lateral
    derivative_heading = float(gradient @ (axes * np.array([projections[1], -projections[0]])))
    return radius, grad_normal, derivative_heading


class RearEscapeCBF(FSMHOCBF):
    """Separation-target FSM with an optional experimental rear CBF."""

    def __init__(self, config=None):
        super().__init__(config)
        self._clear_rear_state()

    def _clear_rear_state(self):
        self._rear_axes = {}
        self._rear_modes = {}
        self._escape_target = None
        self._escape_lateral_target = None
        self._escape_axis = None
        self._escape_attacker = None
        self._escape_complete_corridor = False
        self._step_rears = None
        self._rear_recovery_pending = False
        self._rear_longitudinal_hold = False
        self._rear_guard_release = False

    def filter(self, nominal_control, ego, obstacles, road):
        self._step_rears = None
        self._step_rears = self._rear_candidates(ego, obstacles, road)
        try:
            return super().filter(nominal_control, ego, obstacles, road)
        finally:
            self._step_rears = None

    def reset(self):
        super().reset()
        self._clear_rear_state()

    def _vehicle_barrier(self, ego, other, road):
        barrier = super()._vehicle_barrier(ego, other, road)
        gain = self.config.rear_hocbf_gain
        if (gain <= 0. or other.kind != 'vehicle'
                or float((other.position-ego.position)@road.tangent) >= 0.):
            return barrier
        d = barrier.diagnostic
        lower = (barrier.constraint.lower
                 + (self.config.alpha1+self.config.alpha2-2.*gain)*d.h_dot
                 + (self.config.alpha1*self.config.alpha2-gain**2)*d.h)
        return _Barrier(replace(barrier.constraint, lower=lower),
                        replace(d, psi1=d.h_dot+gain*d.h))

    def _rear_candidates(self, ego, obstacles, road):
        if self._step_rears is not None:
            names = {other.name for other in obstacles}
            return [other for other in self._step_rears if other.name in names]
        result = []
        for other in obstacles:
            if other.kind != "vehicle":
                continue
            if self.config.rear_require_aligned_heading and other.name not in self._rear_axes:
                error = np.arctan2(np.sin(other.heading-road.heading), np.cos(other.heading-road.heading))
                if abs(error) > np.pi/4.:
                    # A crossing/oncoming actor behind the ego is not a
                    # following rear attacker. Its generic vehicle CBF remains.
                    continue
            relative = other.position - ego.position
            if np.linalg.norm(relative) > self.config.sensor_range:
                continue
            longitudinal = float(relative @ road.tangent)
            ego_long = _projected_half_extents(ego, road.tangent, road.left_normal)[0]
            other_long = _projected_half_extents(other, road.tangent, road.left_normal)[0]
            # Center-behind is not enough: an adjacent car with almost the
            # same longitudinal position is a SIDE interaction, not an attacker
            # whose width should define the rear escape target.
            if longitudinal >= -(ego_long + other_long) and other.name not in self._rear_axes:
                continue
            diagnostic = self._vehicle_barrier(ego, other, road).diagnostic
            if (diagnostic.h <= self.config.fsm_exit_h or diagnostic.psi1 <= self.config.fsm_exit_psi1
                    or self._rear_ttc(ego, other, road) <= 1.2*self.config.rear_trigger_ttc):
                _, ego_lat = _projected_half_extents(ego, road.tangent, road.left_normal)
                _, other_lat = _projected_half_extents(other, road.tangent, road.left_normal)
                slow_near_rear = (self.config.rear_guard_allow_slow_approach and ego.speed < 3.
                                 and -longitudinal < 15.
                                 and abs(float(relative@road.left_normal)) <= ego_lat+other_lat+1.)
                if slow_near_rear or self._current_lateral_overlap(ego, other, road) or self._non_overlapping_vehicle_is_converging(
                    ego, other, road, envelope_h_dot=diagnostic.h_dot
                ):
                    result.append(other)
        return result

    def _rear_ttc(self, ego, other, road):
        """Instantaneous bumper gap / closing speed, not a trajectory rollout."""
        if self.config.rear_trigger_ttc <= 0.:
            return float('inf')
        relative = other.position-ego.position
        longitudinal = float(relative@road.tangent)
        extent = (_projected_half_extents(ego, road.tangent, road.left_normal)[0]
                  + _projected_half_extents(other, road.tangent, road.left_normal)[0])
        closing = float((other.velocity-ego.velocity)@road.tangent)
        if longitudinal >= 0. or closing <= .1:
            return float('inf')
        return max(0., -longitudinal-extent)/closing

    def _rear_entering(self, ego, other, road, diagnostic):
        return (diagnostic.h <= self.config.fsm_enter_h
                or diagnostic.psi1 <= self.config.fsm_enter_psi1
                or self._rear_ttc(ego, other, road) <= self.config.rear_trigger_ttc)

    def _escape_displacement(self, direction, ego, rears, road):
        # Terminal geometry assumes the ego has returned parallel to the road;
        # the online CBF includes the actual heading-dependent support instead.
        required = self._required_lateral_displacement(direction, ego, rears, road)
        preferred = (self.config.fsm_lateral_target if self.config.rear_use_native_dynamics
                     else self.config.rear_escape_target_buffer)
        return float(max(required, preferred))

    def _required_lateral_displacement(self, direction, ego, rears, road):
        ego_radius = 2.0**.25*ego.width/2.
        return max([0., *[direction*float((other.position-ego.position)@road.left_normal)
                          + ego_radius+footprint_support(other, road.left_normal)[0]
                          + self.config.lateral_margin+self.config.rear_escape_target_buffer
                          for other in rears]])

    def _bounded_target(self, direction, ego, rears, road):
        required = self._escape_displacement(direction, ego, rears, road)
        clearance = road.left_clearance if direction > 0 else road.right_clearance
        if clearance is None or not np.isfinite(clearance):
            return required, required, 1.0
        available = float(clearance) - ego.width / 2.0 - self.config.road_margin
        # A partial displacement can be useful WITH longitudinal acceleration.
        # It is explicitly not certified as a complete lateral escape corridor.
        target = min(required, 0.85 * max(available, 0.0))
        if target < min(required, self.config.rear_min_lateral_shift):
            return required, target, -1.0
        return required, target, (available - target) / max(ego.width, 1.0)

    def _direction_margin(self, direction, ego, obstacles, road):
        rears = self._rear_candidates(ego, obstacles, road)
        if not rears:
            return super()._direction_margin(direction, ego, obstacles, road)
        required, target, score = self._bounded_target(direction, ego, rears, road)
        current_direction = (1. if self.supervisor.state is FSMState.ESCAPE_LEFT else
                             -1. if self.supervisor.state is FSMState.ESCAPE_RIGHT else 0.)
        if (self.config.rear_latched_corridor and direction == current_direction
                and self._escape_target is not None):
            # Assess the REMAINING path to the fixed target. Requiring a new
            # >=0.75 m displacement each frame falsely cancels an ongoing escape.
            if self._escape_lateral_target is not None and road.reference_lateral_error is not None:
                target = max(0., direction*(self._escape_lateral_target+road.reference_lateral_error))
            else:
                target = max(0., direction*float((self._escape_target-ego.position)@road.left_normal))
            clearance = road.left_clearance if direction > 0 else road.right_clearance
            available = (float('inf') if clearance is None else
                         clearance-ego.width/2.-self.config.road_margin)
            score = (available-target)/max(ego.width, 1.)
        if score < 0.0:
            return score
        # A terminal escape position is evaluated with road-aligned ego heading.
        # Current rotating corners are independently protected by the road CBF.
        terminal_ego = replace(ego, heading=road.heading)
        scores = [score]
        virtual_ego = replace(terminal_ego, position=ego.position + direction * target * road.left_normal)
        for other in obstacles:
            # Reject a currently occupied lateral corridor, not just its end.
            rel = other.position - ego.position
            longitudinal = float(rel @ road.tangent)
            side = direction * float(rel @ road.left_normal)
            ego_long, ego_lat = _projected_half_extents(ego, road.tangent, road.left_normal)
            other_long, other_lat = _projected_half_extents(other, road.tangent, road.left_normal)
            if (abs(longitudinal) <= ego_long + other_long + self.config.longitudinal_margin
                    and side > 0.0 and side - ego_lat - other_lat < target):
                return -1.0
            if (self.config.rear_latched_corridor and other.kind == 'vehicle'
                    and longitudinal < -(ego_long+other_long+self.config.longitudinal_margin)):
                # A car still behind the swept lateral corridor is a threat,
                # not a static blockage AT the target. Its CBF remains in the QP.
                # Testing a hypothetical endpoint against this rear envelope
                # otherwise cancels the very maneuver needed to separate.
                continue
            diagnostic = self._vehicle_barrier(virtual_ego, other, road).diagnostic
            scores.append(diagnostic.h)
        margin = float(min(scores))
        if self.config.rear_latched_corridor and margin >= 0.:
            actual_required = self._required_lateral_displacement(direction, ego, rears, road)
            # Prefer a side that can separate actual vehicle footprints over
            # a wider-but-incomplete shift toward an offset rear attacker.
            if target >= actual_required-self.config.constraint_tolerance:
                margin += 1.
        return margin

    def _prepare_maneuver(self, ego, obstacles, road, decision):
        rears = self._rear_candidates(ego, obstacles, road)
        for other in rears:
            self._rear_axes.setdefault(other.name, road.left_normal.copy())
        names = {other.name for other in obstacles}
        self._rear_axes = {k: v for k, v in self._rear_axes.items() if k in names}
        self._rear_modes = {k: v for k, v in self._rear_modes.items() if k in names}
        escaping = decision.state in (FSMState.ESCAPE_LEFT, FSMState.ESCAPE_RIGHT)
        if (self.config.rear_return_to_reference and escaping
                and road.reference_lateral_error is not None):
            self._rear_recovery_pending = True
        elif (self._rear_recovery_pending and road.reference_lateral_error is not None
              and abs(road.reference_lateral_error) < .2
              and abs(float(ego.velocity@road.left_normal)) < .3):
            self._rear_recovery_pending = False
        if escaping and (decision.transitioned or self._escape_target is None) and rears:
            direction = 1.0 if decision.state is FSMState.ESCAPE_LEFT else -1.0
            required, distance, _ = self._bounded_target(direction, ego, rears, road)
            self._escape_complete_corridor = distance >= required - 1e-6
            self._escape_axis = road.left_normal.copy()
            self._escape_target = ego.position + direction * distance * self._escape_axis
            self._escape_lateral_target = (
                -road.reference_lateral_error+direction*distance
                if self.config.rear_frenet_escape and road.reference_lateral_error is not None else None)
            self._escape_attacker = min(rears, key=lambda r: np.linalg.norm(r.position - ego.position)).name
        elif not escaping:
            self._escape_target = self._escape_axis = self._escape_attacker = None
            self._escape_lateral_target = None
            self._escape_complete_corridor = False

    def _assess_threats(self, ego, active_obstacles, sensor_obstacles, road, nominal_control):
        assessment = super()._assess_threats(ego, active_obstacles, sensor_obstacles, road, nominal_control)
        rears = self._rear_candidates(ego, active_obstacles, road)
        rear_diagnostics = [self._vehicle_barrier(ego, other, road).diagnostic for other in rears]
        assessment = replace(
            assessment,
            rear_threat=any(self._rear_entering(ego, other, road, b)
                            for other, b in zip(rears, rear_diagnostics)),
            rear_clear=all(b.h >= self.config.fsm_exit_h and b.psi1 >= self.config.fsm_exit_psi1
                           and self._rear_ttc(ego, other, road) > 1.2*self.config.rear_trigger_ttc
                           for other, b in zip(rears, rear_diagnostics)),
        )
        self._rear_longitudinal_hold = self._longitudinal_hold_is_feasible(
            ego, rears, active_obstacles, road)
        if self._rear_longitudinal_hold:
            # This only suppresses an unnecessary lateral intent. Every active
            # vehicle and road CBF remains in the same downstream safety QP.
            assessment = replace(assessment, rear_threat=False, rear_clear=True)
        released = self.config.rear_release_completed_escape and self._rear_guard_release
        if released:
            assessment = replace(assessment, rear_threat=False, rear_clear=True)
        if self._escape_target is not None:
            direction = 1.0 if self.supervisor.state is FSMState.ESCAPE_LEFT else -1.0
            if self._escape_lateral_target is not None and road.reference_lateral_error is not None:
                remaining = direction*(self._escape_lateral_target+road.reference_lateral_error)
            else:
                remaining = direction * float((self._escape_target - ego.position) @ self._escape_axis)
            approaching = any(other.name == self._escape_attacker
                              for other in self._rear_candidates(ego, sensor_obstacles, road))
            if (approaching and remaining > self.config.rear_escape_target_buffer
                    and not self._rear_longitudinal_hold):
                if not released:
                    assessment = replace(assessment, rear_threat=True, rear_clear=False)
        return assessment

    def _longitudinal_hold_is_feasible(self, ego, rears, obstacles, road):
        """Algebraic current-state input interval, not action search or rollout.

        Check a road-aligned steering feedback and intersect all active CBF
        half-spaces along the acceleration axis. Require an initial CBF-safe
        rear state, a closing-time buffer and unused acceleration authority.
        This is NOT a finite-horizon viability or reachability guarantee.
        """
        if not self.config.rear_prefer_longitudinal_viability or not rears:
            return False
        # Do not abandon an already committed escape just because a single
        # current-state acceleration interval reopens during the maneuver.
        if self.supervisor.state in (FSMState.ESCAPE_LEFT, FSMState.ESCAPE_RIGHT, FSMState.RECOVERY):
            return False
        reserve_upper = self.config.max_acceleration*(1.-self.config.rear_longitudinal_reserve_fraction)
        for other in rears:
            d = self._vehicle_barrier(ego, other, road).diagnostic
            if d.h < 0. or d.psi1 < 0. or self._rear_ttc(ego, other, road) <= self.config.rear_hold_min_ttc:
                return False
            if float(other.acceleration@road.tangent)+self.config.other_accel_uncertainty > reserve_upper:
                return False
        heading_error = np.arctan2(np.sin(ego.heading-road.heading), np.cos(ego.heading-road.heading))
        curvature = float(np.clip(-self.config.fsm_recovery_heading_gain*heading_error/max(ego.speed, 1.),
                                  -self.config.max_curvature, self.config.max_curvature))
        barriers = [self._vehicle_barrier(ego, other, road) for other in obstacles]
        if self.config.enforce_road_barriers:
            barriers.extend(self._road_barriers(ego, road))
        lower, upper = self.config.min_acceleration, self.config.max_acceleration
        for b in barriers:
            ca, ck = b.constraint.coefficients
            residual_lower = float(b.constraint.lower-ck*curvature)
            if abs(ca) < 1e-10:
                if residual_lower > self.config.constraint_tolerance:
                    return False
            elif ca > 0.:
                lower = max(lower, residual_lower/ca)
            else:
                upper = min(upper, residual_lower/ca)
        return lower <= min(upper, reserve_upper)-self.config.constraint_tolerance

    def _current_active_obstacles(self, ego, obstacles, road):
        active = super()._current_active_obstacles(ego, obstacles, road)
        names = {other.name for other in active}
        # A rear attacker remains covered while passing alongside; do not
        # replace its constraint with a centered front HOCBF mid-maneuver.
        for other in obstacles:
            if (self.config.rear_trigger_ttc > 0. and other.name not in names
                    and self._rear_candidates(ego, [other], road)):
                active.append(other)
                names.add(other.name)
            if other.name in self._rear_axes and other.name not in names:
                relative = other.position - ego.position
                if (np.linalg.norm(relative) <= self.config.sensor_range
                        and abs(float(relative @ road.tangent)) <= ego.length + other.length):
                    active.append(other)
        return active

    def _build_barriers(self, ego, obstacles, road):
        if not self.config.rear_velocity_barrier_enabled:
            return super()._build_barriers(ego, obstacles, road)
        barriers = []
        state = self.supervisor.state
        mode = 1.0 if state is FSMState.ESCAPE_LEFT else -1.0 if state is FSMState.ESCAPE_RIGHT else 0.0
        for other in obstacles:
            if other.name not in self._rear_axes:
                barriers.append(self._vehicle_barrier(ego, other, road))
                continue
            axis = self._rear_axes[other.name]
            candidate = self._rear_support_barrier(ego, other, axis, mode)
            previous_mode = self._rear_modes.get(other.name, 0.0)
            if previous_mode != mode and candidate.diagnostic.h < 0.0:
                previous = self._rear_support_barrier(ego, other, axis, previous_mode)
                if previous.diagnostic.h >= 0.0:
                    # A mode switch must not discard an already valid safe set.
                    candidate = previous
                    self._rear_modes[other.name] = previous_mode
                else:
                    self._rear_modes[other.name] = mode  # explicitly out-of-set recovery
            else:
                self._rear_modes[other.name] = mode
            barriers.append(candidate)  # REPLACE, do not also impose the old rear HOCBF
        if self.config.enforce_road_barriers:
            barriers.extend(self._road_barriers(ego, road))
        return barriers

    def _road_barriers(self, ego, road):
        if self.config.rear_velocity_road_barriers:
            return self._velocity_corner_road_barriers(ego, road)
        if self.config.rear_use_native_dynamics:
            return self._corner_road_barriers(ego, road)
        return super()._road_barriers(ego, road)

    def _velocity_corner_road_barriers(self, ego, road):
        """Current-state road CBF with explicit inward-velocity margin.

        h = corner_gap + tau(v) * smooth_min(inward_velocity, 0).
        h <= corner_gap, so a nonnegative h places the actual corner inside.
        Both the corner-yaw term and the derivative of tau(v) are included.
        No trajectory rollout or reachable-set computation is used.
        """
        heading = ego.heading_vector
        lateral = np.array([-heading[1], heading[0]])
        speed = ego.speed
        denominator = np.sqrt(speed**2 + .1**2)
        tau = ego.length/denominator + .2
        barriers = []
        for side_name, clearance, supplied_normal, default_normal in (
            ('road:left', road.left_clearance, road.left_inward_normal, -road.left_normal),
            ('road:right', road.right_clearance, road.right_inward_normal, road.left_normal),
        ):
            if clearance is None or not np.isfinite(clearance):
                continue
            gradient = default_normal if supplied_normal is None else np.asarray(supplied_normal)
            # Clearances are along the road transverse axis; a fitted, oblique
            # physical boundary uses its perpendicular distance in the CBF.
            perpendicular_clearance = clearance*float(gradient@default_normal)
            for end_name, end in [('front', 1.), ('rear', -1.)]:
                for edge_name, edge in [('left', 1.), ('right', -1.)]:
                    offset = end*.5*ego.length*heading + edge*.5*ego.width*lateral
                    rotated = np.array([-offset[1], offset[0]])
                    name = f'{side_name}:{end_name}_{edge_name}'
                    local_gradient = gradient
                    gap = float(perpendicular_clearance + gradient@offset - self.config.road_margin)
                    if road.corner_boundaries and name in road.corner_boundaries:
                        sensed_gap, local_gradient = road.corner_boundaries[name]
                        local_gradient = np.asarray(local_gradient)
                        gap = float(sensed_gap-self.config.road_margin)
                    # Each corner may see a differently oriented segment at a bend.
                    corner_inward = float(local_gradient@ego.velocity)
                    corner_smooth = np.sqrt(corner_inward**2+1e-6)
                    corner_negative = .5*(corner_inward-corner_smooth)
                    corner_negative_derivative = .5*(1.-corner_inward/corner_smooth)
                    h = gap + tau*corner_negative
                    acceleration_coefficient = (tau*corner_negative_derivative*float(local_gradient@heading)
                        - ego.length*corner_negative/denominator**3*float(ego.velocity@heading))
                    curvature_coefficient = (speed*float(local_gradient@rotated)
                        + tau*corner_negative_derivative*speed**2*float(local_gradient@lateral)
                        - ego.length*corner_negative/denominator**3*speed**2*float(ego.velocity@lateral))
                    coefficients = np.array([acceleration_coefficient, curvature_coefficient])
                    barriers.append(_Barrier(
                        constraint=LinearConstraint(coefficients,
                            -corner_inward-self.config.alpha1*h, name, 'road'),
                        diagnostic=BarrierDiagnostic(name=name, kind='road', h=h,
                            h_dot=corner_inward, psi1=corner_inward+self.config.alpha1*h,
                            barrier_model=ROAD_MODEL, geometric_gap=gap,
                            steering_coefficient=float(curvature_coefficient)),
                    ))
        return barriers

    def _rear_support_barrier(self, ego, other, axis, mode):
        """h = n.p - support_e - support_r - margin - min(n.v,0)^2/(2s).

        n = normalize(p + mode*bias*w(p)*axis). The frozen axis is latched when the
        rear threat is acquired; n itself moves and its exact derivative is
        included. This is a current-state algebraic map, NOT a rollout.
        """
        p = ego.position - other.position
        relative_velocity = ego.velocity - other.velocity
        tangent = np.array([axis[1], -axis[0]])
        longitudinal = float(p @ tangent)
        scale_sq = self.config.rear_normal_bias ** 2
        weight = longitudinal ** 2 / (longitudinal ** 2 + scale_sq)
        weight_rate = (2.0 * longitudinal * scale_sq / (longitudinal ** 2 + scale_sq) ** 2
                       * float(relative_velocity @ tangent))
        shifted = p + mode * self.config.rear_normal_bias * weight * axis
        shifted_rate = relative_velocity + mode * self.config.rear_normal_bias * weight_rate * axis
        # The bias vanishes alongside, so the only remaining singularity is
        # overlapping vehicle centers, already outside the collision-free set.
        distance = max(float(np.linalg.norm(shifted)), 1e-6)
        normal = shifted / distance
        normal_rate = (np.eye(2) - np.outer(normal, normal)) @ shifted_rate / distance
        ego_radius, ego_gradient, ego_heading_gradient = footprint_support(ego, normal)
        rear_radius, rear_gradient, rear_heading_gradient = footprint_support(other, normal)
        gap = float(normal @ p - ego_radius - rear_radius - self.config.lateral_margin)
        closing_projection = float(normal @ relative_velocity)
        closing = min(closing_projection, 0.0)
        ratio = closing / self.config.rear_relative_accel_scale
        h = gap - closing ** 2 / (2.0 * self.config.rear_relative_accel_scale)
        heading = ego.heading_vector
        lateral = np.array([-heading[1], heading[0]])
        coefficients = np.array([
            -ratio * float(normal @ heading),
            -ego_heading_gradient * ego.speed - ratio * ego.speed ** 2 * float(normal @ lateral),
        ])
        drift = float(
            normal_rate @ (p - ego_gradient - rear_gradient) + closing_projection
            - rear_heading_gradient * other.yaw_rate
            - ratio * (normal_rate @ relative_velocity - normal @ other.acceleration)
        )
        robust_loss = (abs(ratio) * self.config.other_accel_uncertainty
                       + abs(rear_heading_gradient) * self.config.rear_yaw_rate_uncertainty)
        name = f"{other.kind}:{other.name}"
        return _Barrier(
            constraint=LinearConstraint(coefficients, -drift - self.config.alpha1 * h + robust_loss,
                                        name, "vehicle"),
            diagnostic=BarrierDiagnostic(name=name, kind="vehicle", h=h, h_dot=drift,
                                         psi1=drift + self.config.alpha1 * h,
                                         barrier_model=REAR_MODEL, geometric_gap=gap,
                                         steering_coefficient=float(coefficients[1])),
        )

    def _reference_for_state(self, state, nominal, ego, road, assessment=None):
        if (state in (FSMState.ESCAPE_LEFT, FSMState.ESCAPE_RIGHT)
                and self._escape_target is not None):
            frenet = self._escape_lateral_target is not None and road.reference_lateral_error is not None
            axis = road.left_normal if frenet else self._escape_axis
            error = (self._escape_lateral_target+road.reference_lateral_error if frenet else
                     float((self._escape_target - ego.position) @ axis))
            lateral_speed = float(ego.velocity @ axis)
            lateral_acceleration = (self.config.rear_escape_position_gain * error
                                    - self.config.rear_escape_damping * lateral_speed)
            left = np.array([-ego.heading_vector[1], ego.heading_vector[0]])
            authority = max(float(left @ axis), 0.20)
            feedforward = float(road.reference_curvature or 0.) if frenet else 0.
            curvature = feedforward+lateral_acceleration / (max(ego.speed ** 2, 1.0) * authority)
            limit = self.config.max_curvature * self.config.escape_curvature_fraction
            return Control2D(nominal.acceleration, float(np.clip(curvature, -limit, limit)))
        if (self.config.rear_return_to_reference and self._rear_recovery_pending
                and road.reference_lateral_error is not None
                and assessment is not None and assessment.risks_clear):
            lateral_speed = float(ego.velocity@road.left_normal)
            lateral_acceleration = float(np.clip(self.config.rear_escape_position_gain*road.reference_lateral_error
                                                - self.config.rear_escape_damping*lateral_speed, -2., 2.))
            left = np.array([-ego.heading_vector[1], ego.heading_vector[0]])
            authority = max(float(left@road.left_normal), .2)
            feedforward = float(road.reference_curvature or 0.) if self.config.rear_frenet_escape else 0.
            curvature = feedforward+lateral_acceleration/(max(ego.speed**2, 1.)*authority)
            return Control2D(nominal.acceleration,
                             float(np.clip(curvature, -self.config.max_curvature, self.config.max_curvature)))
        return super()._reference_for_state(state, nominal, ego, road, assessment)

    def _allow_inactive_passthrough(self):
        return not (self.config.rear_return_to_reference and self._rear_recovery_pending)

    def _executed_diagnostics(self, barriers, control):
        diagnostics = super()._executed_diagnostics(barriers, control)
        for index, barrier in enumerate(barriers):
            if barrier.diagnostic.barrier_model in (REAR_MODEL, ROAD_MODEL):
                h_dot = barrier.diagnostic.h_dot + float(barrier.constraint.coefficients @ control.as_array())
                diagnostics[index] = replace(
                    diagnostics[index], h_dot=h_dot,
                    psi1=h_dot + self.config.alpha1 * barrier.diagnostic.h,
                    barrier_model=barrier.diagnostic.barrier_model, geometric_gap=barrier.diagnostic.geometric_gap,
                    steering_coefficient=barrier.diagnostic.steering_coefficient,
                )
        return diagnostics
