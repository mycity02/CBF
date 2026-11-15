from dataclasses import replace
from typing import Any, Dict, List, Sequence, Tuple
from types import SimpleNamespace

import numpy as np

from bmt.runtime_assurance.cbf2d import TwoDimensionalCBF
from bmt.runtime_assurance.fsm import FSMHOCBF
from bmt.runtime_assurance.metadrive_plugin import MetaDriveCBFPlugin
from bmt.runtime_assurance.types import CBFConfig, Control2D, RoadGeometry, VehicleState2D


class PredictiveEscapeCBF(TwoDimensionalCBF):
    """Two-dimensional HOCBF with predictive left/right escape references."""

    def __init__(self, config: CBFConfig = None):
        super().__init__(config)
        self._candidate_ego = None
        self._candidate_road = None

    def filter(
        self,
        nominal_control: Control2D,
        ego: VehicleState2D,
        obstacles: Sequence[VehicleState2D],
        road: RoadGeometry,
    ):
        self._candidate_ego = ego
        self._candidate_road = road
        try:
            return super().filter(nominal_control, ego, obstacles, road)
        finally:
            self._candidate_ego = None
            self._candidate_road = None

    def _candidate_references(self, nominal: Control2D) -> List[Tuple[str, Control2D]]:
        ego = self._candidate_ego
        road = self._candidate_road
        if ego is None or road is None:
            return super()._candidate_references(nominal)

        horizon = self.config.preview_horizon
        average_speed = max(ego.speed + 0.5 * nominal.acceleration * horizon, 1.0)
        curvature_limit = self.config.max_curvature * self.config.escape_curvature_fraction

        def magnitude(clearance: float) -> float:
            if clearance is None:
                lateral_target = 3.0
            else:
                available = max(
                    0.0,
                    clearance - ego.width / 2.0 - self.config.road_margin,
                )
                lateral_target = min(3.0, 0.70 * available)
            # lateral displacement ~= 0.5 * v^2 * curvature * horizon^2
            value = 2.0 * lateral_target / max(average_speed ** 2 * horizon ** 2, 1e-6)
            return float(np.clip(value, 0.0, curvature_limit))

        left_curvature = max(magnitude(road.left_clearance), nominal.curvature)
        right_curvature = min(-magnitude(road.right_clearance), nominal.curvature)
        return [
            ("nominal", nominal),
            ("left_escape", Control2D(nominal.acceleration, left_curvature)),
            ("right_escape", Control2D(nominal.acceleration, right_curvature)),
        ]


class AdaptiveMetaDriveCBFPlugin(MetaDriveCBFPlugin):
    """Public plug-and-play adapter used by the isolated evaluation entrypoint."""

    name = "predictive_escape_cbf2d"

    def __init__(self, config: CBFConfig = None):
        super().__init__(config)
        self.filter = PredictiveEscapeCBF(self.config)

    def filter_action(
        self,
        env: Any,
        nominal_action: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        nominal_action = np.asarray(nominal_action, dtype=float).reshape(-1)
        if nominal_action.size != 2:
            raise ValueError(f"MetaDrive CBF plugin expects a two-element action, got {nominal_action}")

        ego_object = self._ego_vehicle(env)
        ego = self._vehicle_state(ego_object)
        vehicle_objects = self._vehicle_objects(env)
        obstacles = [
            self._vehicle_state(vehicle)
            for vehicle in vehicle_objects
            if vehicle is not ego_object and str(vehicle.name) != str(ego_object.name)
        ]
        road = self._road_geometry(ego_object)
        nominal_control, actual_max_curvature = self._action_to_control(nominal_action, ego_object)

        active_config = self._runtime_config(actual_max_curvature, ego_object)
        if self.filter.config != active_config:
            self.filter = PredictiveEscapeCBF(active_config)

        result = self.filter.filter(nominal_control, ego, obstacles, road)
        safe_action = self._control_to_action(
            result.control,
            ego_object,
            actual_max_curvature,
        )
        telemetry = result.to_telemetry()
        telemetry.update(
            {
                "rta_plugin": self.name,
                "rta_episode_step": self._episode_steps,
                "rta_nominal_action": nominal_action.tolist(),
                "rta_safe_action": safe_action.tolist(),
                "rta_obstacle_count": len(obstacles),
                "rta_road_margin": active_config.road_margin,
                "rta_road_geometry_source": self._road_geometry_source,
            }
        )
        self._remember_velocities([ego_object, *vehicle_objects])
        self._episode_steps += 1
        return safe_action, telemetry

class FSMHOCBFPlugin(MetaDriveCBFPlugin):
    """FSM-supervised single-QP HOCBF adapter for MetaDrive evaluation."""

    name = "fsm_hocbf_qp"

    def _make_filter(self, config: CBFConfig) -> FSMHOCBF:
        return FSMHOCBF(config)


class RearEscapeCBFPlugin(MetaDriveCBFPlugin):
    """Opt-in rear-protection variant; the legacy FSM plugin is unchanged."""

    name = "fsm_rear_escape_cbf_qp"

    def _make_filter(self, config: CBFConfig):
        from bmt.runtime_assurance.rear_escape import RearEscapeCBF
        return RearEscapeCBF(config)

    def _road_geometry(self, ego):
        navigation = getattr(ego, 'navigation', None)
        if (not self.config.rear_use_physical_road_geometry
                or type(navigation).__name__ != 'TrajectoryNavigation'):
            road = super()._road_geometry(ego)
            if self.config.rear_use_physical_road_geometry:
                # Procedural navigation has a real map lane but no trajectory
                # heading property. Falling back to ego yaw rotates the road
                # frame during a maneuver and can falsely block its fixed target.
                lanes = getattr(navigation, 'current_ref_lanes', None)
                if lanes:
                    try:
                        longitudinal, _ = lanes[0].local_coordinates(ego.position)
                        lane_heading = float(lanes[0].heading_theta_at(longitudinal))
                        if np.isfinite(lane_heading):
                            road = replace(road, heading=lane_heading)
                            self._road_geometry_source += '+map_lane_heading'
                    except (AttributeError, TypeError, ValueError):
                        pass
            return road
        # ScenarioEnv's IDM reference is a synthetic width=2 m polyline, NOT
        # the map's drivable road. Its dist_to_*_side must not cap an escape.
        heading = getattr(navigation, 'current_heading_theta_at_long', ego.heading_theta)
        heading = float(heading) if heading is not None and np.isfinite(heading) else float(ego.heading_theta)
        left, right = self._side_detector_clearances(ego, heading, include_clear_range=True)
        if left is None or right is None:
            road = super()._road_geometry(ego)
            self._road_geometry_source += '+physical_sensor_unavailable_fallback'
            return road
        self._road_geometry_source = 'continuous_line_sidewalk_sensor'
        road = (self._fitted_road_geometry(ego, heading, left, right)
                if self.config.rear_fit_boundary_normals else RoadGeometry(heading, left, right))
        if self.config.rear_corner_boundary_sensing:
            road = self._corner_boundary_geometry(ego, road)
        if self.config.rear_return_to_reference or self.config.rear_frenet_escape:
            try:
                trajectory = navigation.reference_trajectory
                longitudinal, _ = trajectory.local_coordinates(ego.position)
                center = trajectory.position(longitudinal, 0.)
                error = float((np.asarray(center)[:2]-np.asarray(ego.position)[:2])@road.left_normal)
                road = replace(road, reference_lateral_error=error)
                if self.config.rear_frenet_escape:
                    # Local spatial geometry, not a future-state rollout.
                    lo, hi = max(0., longitudinal-1.), min(float(trajectory.length), longitudinal+1.)
                    if hi-lo > 1e-6:
                        delta = float(trajectory.heading_theta_at(hi)-trajectory.heading_theta_at(lo))
                        curvature = float(np.arctan2(np.sin(delta), np.cos(delta))/(hi-lo))
                        road = replace(road, reference_curvature=curvature)
            except (AttributeError, TypeError, ValueError):
                pass
        return road

    def _corner_boundary_geometry(self, ego, road):
        """Survey the physical lines at each body corner, including curved roads.

        A center ray minus half-width assumes a straight parallel road. Native
        ScenarioEnv terminates on a line touching ANY corner, so that surrogate
        can miss an approaching bend even with a feasible road QP.
        """
        params = ego.config['side_detector']
        count, distance = int(params['num_lasers']), float(params['distance'])
        detector = ego.side_detector
        e = np.array([np.cos(ego.heading_theta), np.sin(ego.heading_theta)])
        l = np.array([-e[1], e[0]])
        angles = float(ego.heading_theta)+float(detector.start_phase_offset)+2.*np.pi*np.arange(count)/count
        units = np.column_stack((np.cos(angles), np.sin(angles)))
        corners = {}
        for end_name, end in [('front', 1.), ('rear', -1.)]:
            for edge_name, edge in [('left', 1.), ('right', -1.)]:
                offset = end*.5*ego.LENGTH*e+edge*.5*ego.WIDTH*l
                probe = SimpleNamespace(position=np.asarray(ego.position)[:2]+offset,
                                        heading_theta=ego.heading_theta, dynamic_nodes=ego.dynamic_nodes)
                try:
                    result = detector.perceive(probe, physics_world=ego.engine.physics_world.static_world,
                                              num_lasers=count, distance=distance, show=False)
                    fractions = np.asarray(result.cloud_points, dtype=float)
                except Exception:
                    continue  # Missing corner survey retains the center-derived constraint.
                points = distance*fractions[:, None]*units
                for side, sign in [('left', 1.), ('right', -1.)]:
                    target = road.heading+sign*np.pi/2.
                    errors = np.abs(np.arctan2(np.sin(angles-target), np.cos(angles-target)))
                    valid = np.isfinite(fractions) & (fractions > 0.) & (fractions < .999)
                    mask = valid & (errors <= np.pi/3.)
                    inward = -sign*road.left_normal
                    plane = self._fit_side_plane(points[mask], inward)
                    if plane is not None:
                        clearance, normal = plane
                        gap = clearance*float(normal@inward)
                    else:
                        mask = valid & (errors <= np.pi/6.+1e-9)
                        if not np.any(mask):
                            continue
                        gap = float(np.min(distance*fractions[mask]*np.cos(errors[mask])))
                        normal = inward
                    corners[f'road:{side}:{end_name}_{edge_name}'] = (gap, normal)
        if corners:
            self._road_geometry_source += '+corner_surveys'
        return replace(road, corner_boundaries=corners)

    @staticmethod
    def _fit_side_plane(points, inward):
        """Local physical line from nearby sensor hits, never a route-width prior."""
        if len(points) < 2:
            return None
        points = np.asarray(points)
        first = points[np.argmin(np.linalg.norm(points, axis=1))]
        distances = np.linalg.norm(points-first, axis=1)
        valid = np.flatnonzero((distances > .15) & (distances < 12.))
        if not len(valid):
            return None
        second = points[valid[np.argmin(distances[valid])]]
        segment = second-first
        normal = np.array([-segment[1], segment[0]])/np.linalg.norm(segment)
        if normal@inward < 0.:
            normal = -normal
        alignment = float(normal@inward)
        distance = -float(normal@first)
        if alignment < .5 or distance <= 0.:
            return None
        return distance/alignment, normal

    def _fitted_road_geometry(self, ego, heading, left, right):
        params = ego.config['side_detector']
        count, distance = int(params['num_lasers']), float(params['distance'])
        detector = ego.side_detector
        try:
            result = detector.perceive(ego, physics_world=ego.engine.physics_world.static_world,
                                      num_lasers=count, distance=distance, show=False)
            fractions = np.asarray(result.cloud_points, dtype=float)
        except Exception:
            return RoadGeometry(heading, left, right)
        angles = float(ego.heading_theta)+float(detector.start_phase_offset)+2.*np.pi*np.arange(count)/count
        points = distance*fractions[:, None]*np.column_stack((np.cos(angles), np.sin(angles)))
        normal = np.array([-np.sin(heading), np.cos(heading)])
        planes = []
        for sign in (1., -1.):
            target_angle = heading+sign*np.pi/2.
            errors = np.abs(np.arctan2(np.sin(angles-target_angle), np.cos(angles-target_angle)))
            mask = (errors <= np.pi/3.) & np.isfinite(fractions) & (fractions > 0.) & (fractions < .999)
            planes.append(self._fit_side_plane(points[mask], -sign*normal))
        if any(p is not None for p in planes):
            self._road_geometry_source += '+fitted_normals'
        return RoadGeometry(heading,
                            left if planes[0] is None else planes[0][0],
                            right if planes[1] is None else planes[1][0],
                            None if planes[0] is None else planes[0][1],
                            None if planes[1] is None else planes[1][1])

    def _wheelbase(self, ego_object):
        if self.config.rear_use_native_dynamics:
            front = getattr(ego_object, "FRONT_WHEELBASE", None)
            rear = getattr(ego_object, "REAR_WHEELBASE", None)
            if front is not None and rear is not None and float(front + rear) > 0.:
                return float(front + rear)
        return super()._wheelbase(ego_object)

    @staticmethod
    def _native_actuator_gains(ego_object):
        """Nominal SI gains for Bullet's per-wheel force / brake impulse API.

        Positive engine input is force; braking input is impulse per physics
        substep, NOT the same unit. Rolling resistance and transient tire slip
        remain model errors, checked separately in the diagnostic fixture.
        """
        config = ego_object.config
        mass = float(ego_object.MASS)
        physics_dt = float(ego_object.engine.global_config['physics_world_step_size'])
        drive = 4.0 * float(config['max_engine_force']) / mass
        brake = 4.0 * float(config['max_brake_force']) / (mass * physics_dt)
        if not np.all(np.isfinite([drive, brake])) or min(drive, brake) <= 0.:
            raise ValueError('Native actuator parameters must be positive and finite')
        coast = -4.0 * 2.0 / (mass * physics_dt)
        return drive, brake, coast

    def _action_to_control(self, action, ego_object):
        nominal, max_curvature = super()._action_to_control(action, ego_object)
        if not self.config.rear_use_native_dynamics:
            return nominal, max_curvature
        drive, brake, coast = self._native_actuator_gains(ego_object)
        throttle = float(np.clip(np.asarray(action)[1], -1., 1.))
        acceleration = coast if abs(throttle) < 1e-12 else throttle * (drive if throttle > 0. else brake)
        return Control2D(acceleration, nominal.curvature), max_curvature

    def _control_to_action(self, control, ego_object, actual_max_curvature):
        action = super()._control_to_action(control, ego_object, actual_max_curvature)
        if self.config.rear_use_native_dynamics:
            drive, brake, coast = self._native_actuator_gains(ego_object)
            if abs(control.acceleration-coast) < 1e-10:
                throttle = 0.
            else:
                throttle = control.acceleration / (drive if control.acceleration >= 0. else brake)
            action[1] = float(np.clip(throttle, -1., 1.))
        return action

    def _runtime_config(self, actual_max_curvature, ego_object=None):
        runtime = super()._runtime_config(actual_max_curvature, ego_object)
        if self.config.rear_use_native_dynamics and ego_object is not None:
            drive, brake, _ = self._native_actuator_gains(ego_object)
            speed = float(np.linalg.norm(np.asarray(ego_object.velocity)[:2]))
            runtime = replace(runtime,
                              max_acceleration=min(runtime.max_acceleration, drive),
                              min_acceleration=max(runtime.min_acceleration, -brake),
                              max_curvature=min(runtime.max_curvature,
                                                runtime.rear_max_lateral_acceleration / max(speed**2, 1.)))
            if runtime.rear_use_acceleration_metric:
                runtime = replace(runtime, curvature_weight=(max(speed, 1.) ** 4
                                  * runtime.rear_lateral_acceleration_weight))
            # Bounds change with speed. Update them WITHOUT resetting the FSM,
            # latched escape target or rear tracking at every control step.
            self.filter.config = runtime
            self.filter.supervisor.config = runtime
        return runtime

    def filter_action(self, env, nominal_action):
        action, telemetry = super().filter_action(env, nominal_action)
        if self.config.rear_use_native_dynamics:
            if not telemetry['rta_intervened']:
                action = np.asarray(nominal_action, dtype=np.float32).copy()
                telemetry['rta_safe_action'] = action.tolist()
            telemetry['rta_native_dynamics'] = True
            telemetry['rta_curvature_bound'] = self.filter.config.max_curvature
            telemetry['rta_acceleration_upper_bound'] = self.filter.config.max_acceleration
            telemetry['rta_actuator_gains'] = list(self._native_actuator_gains(self._ego_vehicle(env)))
        rear = [b for b in telemetry["rta_barriers"]
                if b["barrier_model"] == "rear_velocity_support_cbf"]
        telemetry["rta_rear_barrier_count"] = len(rear)
        telemetry["rta_rear_outside_safe_set_count"] = sum(b["h"] < 0.0 for b in rear)
        telemetry["rta_rear_complete_lateral_corridor"] = bool(self.filter._escape_complete_corridor)
        target = self.filter._escape_target
        telemetry['rta_escape_target'] = None if target is None else target.tolist()
        telemetry['rta_rear_longitudinal_hold'] = bool(self.filter._rear_longitudinal_hold)
        telemetry['rta_escape_lateral_target'] = self.filter._escape_lateral_target
        return action, telemetry
