from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from bmt.runtime_assurance.cbf2d import TwoDimensionalCBF
from bmt.runtime_assurance.types import CBFConfig, Control2D, RoadGeometry, VehicleState2D


class MetaDriveCBFPlugin:
    """Plug-in adapter between MetaDrive's normalized actions and the 2-D CBF.

    This class intentionally uses duck typing and imports no MetaDrive internals.
    It can therefore live outside MetaDrive and be removed without changing the
    simulator or the trained policy.
    """

    name = "cbf2d"

    def __init__(self, config: Optional[CBFConfig] = None):
        self.config = config or CBFConfig()
        self.filter = self._make_filter(self.config)
        self._last_velocities: Dict[str, np.ndarray] = {}
        self._episode_steps = 0
        self._road_geometry_source = "reference_route"

    def reset(self, env: Any) -> None:
        self._last_velocities = {
            str(vehicle.name): np.asarray(vehicle.velocity, dtype=float)[:2].copy()
            for vehicle in self._vehicle_objects(env)
        }
        ego = self._ego_vehicle(env)
        self._last_velocities[str(ego.name)] = np.asarray(ego.velocity, dtype=float)[:2].copy()
        self._episode_steps = 0
        reset_filter = getattr(self.filter, "reset", None)
        if callable(reset_filter):
            reset_filter()

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
        obstacles = [
            self._vehicle_state(vehicle)
            for vehicle in self._vehicle_objects(env)
            if vehicle is not ego_object and str(vehicle.name) != str(ego_object.name)
        ]
        road = self._road_geometry(ego_object)

        nominal_control, actual_max_curvature = self._action_to_control(
            action=nominal_action,
            ego_object=ego_object,
        )
        runtime_config = self._runtime_config(actual_max_curvature, ego_object)
        if self.filter.config != runtime_config:
            self.filter = self._make_filter(runtime_config)

        result = self.filter.filter(
            nominal_control=nominal_control,
            ego=ego,
            obstacles=obstacles,
            road=road,
        )
        safe_action = self._control_to_action(
            control=result.control,
            ego_object=ego_object,
            actual_max_curvature=actual_max_curvature,
        )
        telemetry = result.to_telemetry()
        telemetry.update(
            {
                "rta_plugin": self.name,
                "rta_episode_step": self._episode_steps,
                "rta_nominal_action": nominal_action.tolist(),
                "rta_safe_action": safe_action.tolist(),
                "rta_obstacle_count": len(obstacles),
                "rta_road_margin": runtime_config.road_margin,
                "rta_road_geometry_source": self._road_geometry_source,
            }
        )
        self._remember_velocities([ego_object, *self._vehicle_objects(env)])
        self._episode_steps += 1
        return safe_action, telemetry

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "config": self.config.to_dict()}

    def _make_filter(self, config: CBFConfig) -> TwoDimensionalCBF:
        return TwoDimensionalCBF(config)

    def _runtime_config(self, actual_max_curvature: float, ego_object: Any = None) -> CBFConfig:
        return replace(
            self.config,
            max_curvature=min(actual_max_curvature, self.config.max_curvature),
        )
    def _vehicle_state(self, vehicle: Any) -> VehicleState2D:
        name = str(vehicle.name)
        velocity = np.asarray(vehicle.velocity, dtype=float)[:2]
        previous = self._last_velocities.get(name)
        if previous is None:
            acceleration = np.zeros(2, dtype=float)
        else:
            acceleration = (velocity - previous) / self.config.dt
        acceleration = _clip_vector_norm(acceleration, self.config.estimated_accel_limit)
        return VehicleState2D(
            name=name,
            position=np.asarray(vehicle.position, dtype=float)[:2],
            velocity=velocity,
            acceleration=acceleration,
            heading=float(vehicle.heading_theta),
            length=float(vehicle.LENGTH),
            width=float(vehicle.WIDTH),
            kind=str(getattr(vehicle, "metadrive_type", "vehicle")),
        )

    def _road_geometry(self, ego: Any) -> RoadGeometry:
        navigation = getattr(ego, "navigation", None)
        heading = float(ego.heading_theta)
        if navigation is not None:
            candidate = getattr(navigation, "current_heading_theta_at_long", None)
            if candidate is not None and np.isfinite(candidate):
                heading = float(candidate)
        reference_geometry = RoadGeometry(
            heading=heading,
            left_clearance=_finite_or_none(getattr(ego, "dist_to_left_side", None)),
            right_clearance=_finite_or_none(getattr(ego, "dist_to_right_side", None)),
        )
        self._road_geometry_source = "reference_route"
        if not self.config.road_use_side_detector:
            return reference_geometry

        left, right = self._side_detector_clearances(ego, heading)
        if left is None and right is None:
            return reference_geometry
        self._road_geometry_source = "reference_route+side_detector"
        return RoadGeometry(
            heading=heading,
            left_clearance=_minimum_clearance(
                reference_geometry.left_clearance,
                left,
            ),
            right_clearance=_minimum_clearance(
                reference_geometry.right_clearance,
                right,
            ),
        )

    @staticmethod
    def _side_detector_clearances(
        ego: Any,
        road_heading: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        vehicle_config = getattr(ego, "config", {}) or {}
        detector_config = vehicle_config.get("side_detector", {}) or {}
        num_lasers = int(detector_config.get("num_lasers", 0))
        distance = float(detector_config.get("distance", 0.0))
        detector = getattr(ego, "side_detector", None)
        engine = getattr(ego, "engine", None)
        physics_world = getattr(getattr(engine, "physics_world", None), "static_world", None)
        if detector is None or physics_world is None or num_lasers <= 0 or distance <= 0.0:
            return None, None

        try:
            detected = detector.perceive(
                ego,
                physics_world=physics_world,
                num_lasers=num_lasers,
                distance=distance,
                show=False,
            )
            cloud_points = np.asarray(detected.cloud_points, dtype=float).reshape(-1)
        except Exception:
            return None, None
        if cloud_points.size != num_lasers:
            return None, None

        start_phase = float(getattr(detector, "start_phase_offset", np.pi / 2.0))
        relative_angles = start_phase + 2.0 * np.pi * np.arange(num_lasers) / num_lasers
        world_angles = float(ego.heading_theta) + relative_angles

        def clearance(target_angle: float) -> Optional[float]:
            angle_errors = np.abs(
                np.arctan2(
                    np.sin(world_angles - target_angle),
                    np.cos(world_angles - target_angle),
                )
            )
            for index in np.argsort(angle_errors):
                error = float(angle_errors[index])
                if error > np.pi / 3.0:
                    break
                fraction = float(cloud_points[index])
                if np.isfinite(fraction) and 0.0 <= fraction < 0.999:
                    return fraction * distance * max(float(np.cos(error)), 0.0)
            return None

        return (
            clearance(road_heading + np.pi / 2.0),
            clearance(road_heading - np.pi / 2.0),
        )

    def _action_to_control(
        self,
        action: np.ndarray,
        ego_object: Any,
    ) -> Tuple[Control2D, float]:
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))
        if throttle_brake >= 0.0:
            acceleration = throttle_brake * self.config.max_acceleration
        else:
            acceleration = -(-throttle_brake) * abs(self.config.min_acceleration)

        wheelbase = self._wheelbase(ego_object)
        max_angle = self._max_steering_radians(ego_object)
        actual_max_curvature = min(
            abs(float(np.tan(max_angle) / wheelbase)),
            self.config.max_curvature,
        )
        curvature = float(np.tan(steering * max_angle) / wheelbase)
        curvature = float(np.clip(curvature, -actual_max_curvature, actual_max_curvature))
        return Control2D(acceleration=acceleration, curvature=curvature), actual_max_curvature

    def _control_to_action(
        self,
        control: Control2D,
        ego_object: Any,
        actual_max_curvature: float,
    ) -> np.ndarray:
        wheelbase = self._wheelbase(ego_object)
        max_angle = self._max_steering_radians(ego_object)
        curvature = float(np.clip(control.curvature, -actual_max_curvature, actual_max_curvature))
        steering = float(np.arctan(curvature * wheelbase) / max(max_angle, 1e-6))
        if control.acceleration >= 0.0:
            throttle_brake = control.acceleration / max(self.config.max_acceleration, 1e-6)
        else:
            throttle_brake = control.acceleration / max(abs(self.config.min_acceleration), 1e-6)
        return np.asarray(
            [
                np.clip(steering, -1.0, 1.0),
                np.clip(throttle_brake, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _wheelbase(ego_object: Any) -> float:
        # MetaDrive does not expose a single wheelbase property on every
        # vehicle class. The 0.60*length approximation is used only for the
        # action/curvature adapter, not for collision geometry.
        return max(0.60 * float(ego_object.LENGTH), 1.0)

    @staticmethod
    def _max_steering_radians(ego_object: Any) -> float:
        max_steering_degrees = float(getattr(ego_object, "max_steering", 40.0))
        return max(float(np.deg2rad(max_steering_degrees)), 1e-3)

    @staticmethod
    def _ego_vehicle(env: Any) -> Any:
        ego = getattr(env, "agent", None)
        if ego is None:
            raise RuntimeError("The CBF plugin requires a single-agent MetaDrive environment")
        return ego

    def _vehicle_objects(self, env: Any) -> List[Any]:
        engine = getattr(env, "engine", None)
        traffic_manager = getattr(engine, "traffic_manager", None)
        vehicles = getattr(traffic_manager, "vehicles", None)
        vehicle_list = list(vehicles or [])
        if not self.config.include_vulnerable_road_users:
            return vehicle_list

        get_objects = getattr(engine, "get_objects", None)
        if not callable(get_objects):
            return vehicle_list
        try:
            objects = get_objects()
        except Exception:
            return vehicle_list
        candidates = list(objects.values()) if isinstance(objects, dict) else list(objects or [])
        allowed_kinds = {"vehicle", "pedestrian", "cyclist"}
        participants = [
            obj
            for obj in candidates
            if str(getattr(obj, "metadrive_type", "")).lower() in allowed_kinds
            and all(
                hasattr(obj, attribute)
                for attribute in ("name", "position", "velocity", "heading_theta", "LENGTH", "WIDTH")
            )
        ]
        return participants or vehicle_list

    def _remember_velocities(self, vehicles: Iterable[Any]) -> None:
        current: Dict[str, np.ndarray] = {}
        for vehicle in vehicles:
            current[str(vehicle.name)] = np.asarray(vehicle.velocity, dtype=float)[:2].copy()
        self._last_velocities = current


def _minimum_clearance(
    reference: Optional[float],
    detected: Optional[float],
) -> Optional[float]:
    """Fuse route and physical-line clearances without relaxing either one."""

    values = [value for value in (reference, detected) if value is not None]
    return min(values) if values else None


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _clip_vector_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= limit or norm <= 1e-12:
        return vector
    return vector * (limit / norm)
