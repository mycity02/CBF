from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from bmt.runtime_assurance.cbf2d import TwoDimensionalCBF
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
