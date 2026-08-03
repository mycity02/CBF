from types import SimpleNamespace

import numpy as np

from bmt.runtime_assurance.adaptive_plugin import (
    AdaptiveMetaDriveCBFPlugin,
    PredictiveEscapeCBF,
)
from bmt.runtime_assurance.qp import (
    LinearConstraint,
    solve_prioritized_projection,
    solve_weighted_projection,
)
from bmt.runtime_assurance.types import (
    CBFConfig,
    Control2D,
    RoadGeometry,
    VehicleState2D,
)


def vehicle(name, x, y, speed, heading=0.0, kind="vehicle"):
    velocity = speed * np.asarray([np.cos(heading), np.sin(heading)])
    return VehicleState2D(
        name=name,
        position=np.asarray([x, y]),
        velocity=velocity,
        acceleration=np.zeros(2),
        heading=heading,
        length=4.5,
        width=2.0,
        kind=kind,
    )


def test_two_variable_qp_projects_to_safe_halfspace():
    solution = solve_weighted_projection(
        reference=np.asarray([0.0, 0.0]),
        weights=np.asarray([1.0, 1.0]),
        constraints=[
            LinearConstraint(np.asarray([1.0, 1.0]), 1.0, "diagonal", "test"),
        ],
        lower_bounds=np.asarray([-2.0, -2.0]),
        upper_bounds=np.asarray([2.0, 2.0]),
    )

    assert solution.hard_feasible
    np.testing.assert_allclose(solution.control, [0.5, 0.5], atol=1e-8)


def test_prioritized_qp_never_trades_a_road_boundary_for_a_vehicle_constraint():
    road = LinearConstraint(np.asarray([0.0, -1.0]), 0.0, "road:left", "road")
    vehicle_constraint = LinearConstraint(
        np.asarray([0.0, 1.0]), 0.2, "vehicle:side", "vehicle"
    )
    solution = solve_prioritized_projection(
        reference=np.asarray([0.0, 0.3]),
        weights=np.asarray([1.0, 1.0]),
        constraints=[road, vehicle_constraint],
        lower_bounds=np.asarray([-6.0, -0.3]),
        upper_bounds=np.asarray([3.0, 0.3]),
    )

    assert not solution.hard_feasible
    assert solution.mandatory_feasible
    assert solution.fallback_mode == "relaxed_vehicle"
    assert road.residual(solution.control) >= -1e-7
    assert vehicle_constraint.residual(solution.control) < 0.0


def test_distant_laterally_separated_vehicle_does_not_activate():
    cbf = PredictiveEscapeCBF(CBFConfig(other_accel_uncertainty=0.0))
    ego = vehicle("ego", 0.0, 0.0, 10.0)
    obstacle = vehicle("far_right", 27.0, -8.0, 0.0)
    road = RoadGeometry(heading=0.0, left_clearance=5.0, right_clearance=5.0)

    assert cbf._active_obstacles(ego, [obstacle], road) == []
    diagnostic = cbf._vehicle_barrier(ego, obstacle, road).diagnostic
    assert diagnostic.h > 0.0
    assert diagnostic.psi1 > 0.0


def test_aligned_closing_vehicle_activates_before_it_is_near():
    cbf = PredictiveEscapeCBF(CBFConfig(other_accel_uncertainty=0.0))
    ego = vehicle("ego", 0.0, 0.0, 10.0)
    obstacle = vehicle("front", 35.0, 0.0, 0.0)
    road = RoadGeometry(heading=0.0, left_clearance=5.0, right_clearance=5.0)

    assert cbf._active_obstacles(ego, [obstacle], road) == [obstacle]

def test_distant_vulnerable_road_user_uses_a_shorter_activation_horizon():
    cbf = PredictiveEscapeCBF(
        CBFConfig(
            activation_horizon=3.0,
            vulnerable_road_user_activation_horizon=1.5,
            activation_require_current_lateral_overlap=False,
        )
    )
    ego = vehicle("ego", 0.0, 0.0, 10.0)
    road = RoadGeometry(heading=0.0, left_clearance=5.0, right_clearance=5.0)
    vehicle_obstacle = vehicle("vehicle", 35.0, 0.0, 0.0)
    pedestrian = vehicle("pedestrian", 35.0, 0.0, 0.0, kind="pedestrian")

    assert cbf._active_obstacles(ego, [vehicle_obstacle], road) == [vehicle_obstacle]
    assert cbf._active_obstacles(ego, [pedestrian], road) == []


def test_laterally_separated_cross_traffic_stays_preview_only_until_overlap():
    road = RoadGeometry(heading=0.0, left_clearance=5.0, right_clearance=5.0)
    ego = vehicle("ego", 0.0, 0.0, 8.0)
    crossing = vehicle("crossing", 8.0, 5.0, 2.0, heading=-np.pi / 2.0)
    nominal = Control2D(0.0, 0.0)
    base = CBFConfig(
        activation_horizon=1.5,
        activation_require_current_lateral_overlap=False,
    )
    ungated = PredictiveEscapeCBF(base)
    gated = PredictiveEscapeCBF(
        CBFConfig(
            activation_horizon=1.5,
            activation_require_current_lateral_overlap=True,
        )
    )

    assert ungated._active_obstacles(ego, [crossing], road, nominal) == [crossing]
    assert gated._active_obstacles(ego, [crossing], road, nominal) == []

    pedestrian = vehicle(
        "pedestrian",
        8.0,
        5.0,
        2.0,
        heading=-np.pi / 2.0,
        kind="pedestrian",
    )
    assert gated._active_obstacles(ego, [pedestrian], road, nominal) == [pedestrian]


def test_imminent_vehicle_cut_in_bypasses_current_lateral_overlap_gate():
    road = RoadGeometry(heading=0.0, left_clearance=5.0, right_clearance=5.0)
    ego = vehicle("ego", 0.0, 0.0, 8.0)
    cut_in = vehicle("cut_in", 8.0, 5.0, 5.0, heading=-np.pi / 2.0)
    cbf = PredictiveEscapeCBF(
        CBFConfig(
            activation_lateral_gate_horizon=1.0,
            activation_require_current_lateral_overlap=True,
        )
    )

    assert cbf._active_obstacles(ego, [cut_in], road) == [cut_in]


def test_filter_is_minimally_invasive_without_a_threat():
    cbf = PredictiveEscapeCBF(CBFConfig(other_accel_uncertainty=0.0))
    nominal = Control2D(acceleration=0.4, curvature=0.0)
    result = cbf.filter(
        nominal_control=nominal,
        ego=vehicle("ego", 0.0, 0.0, 10.0),
        obstacles=[],
        road=RoadGeometry(heading=0.0, left_clearance=5.0, right_clearance=5.0),
    )

    assert result.mode == "nominal"
    assert result.hard_feasible
    np.testing.assert_allclose(result.control.as_array(), nominal.as_array(), atol=1e-9)


def test_preview_cap_keeps_all_active_threats_and_nearest_context():
    cbf = PredictiveEscapeCBF(CBFConfig(preview_max_obstacles=3))
    ego = vehicle("ego", 0.0, 0.0, 0.0)
    active = [
        vehicle("active_near", 3.0, 0.0, 0.0),
        vehicle("active_far", 20.0, 0.0, 0.0),
    ]
    context = [
        vehicle("inactive_pedestrian", 1.0, 0.0, 0.0, kind="pedestrian"),
        vehicle("nearest_context", 4.0, 0.0, 0.0),
        vehicle("far_context", 5.0, 0.0, 0.0),
    ]

    selected = cbf._scoring_obstacles(ego, active + context, active)

    assert [obstacle.name for obstacle in selected] == [
        "active_near",
        "active_far",
        "nearest_context",
    ]


def test_preview_cap_never_drops_active_threats_when_active_count_exceeds_cap():
    cbf = PredictiveEscapeCBF(CBFConfig(preview_max_obstacles=1))
    ego = vehicle("ego", 0.0, 0.0, 0.0)
    active = [
        vehicle("active_1", 3.0, 0.0, 0.0),
        vehicle("active_2", 4.0, 0.0, 0.0),
    ]

    selected = cbf._scoring_obstacles(ego, active, active)

    assert selected == active


def test_low_speed_recovery_bypasses_only_an_initially_tight_road_margin():
    cbf = PredictiveEscapeCBF(
        CBFConfig(
            passthrough_without_active_obstacle=False,
            road_low_speed_passthrough=2.0,
        )
    )
    nominal = Control2D(3.0, 0.1)
    result = cbf.filter(
        nominal_control=nominal,
        ego=vehicle("ego", 0.0, 0.0, 0.3, heading=0.06),
        obstacles=[],
        road=RoadGeometry(heading=0.0, left_clearance=1.0, right_clearance=3.0),
    )

    assert result.mode == "nominal"
    assert not result.intervened
    assert result.fallback_mode == "low_speed_road_recovery"
    np.testing.assert_allclose(result.control.as_array(), nominal.as_array(), atol=1e-9)

def test_low_speed_recovery_does_not_bypass_a_physical_road_violation():
    cbf = PredictiveEscapeCBF(
        CBFConfig(
            passthrough_without_active_obstacle=False,
            road_low_speed_passthrough=2.0,
        )
    )
    nominal = Control2D(3.0, 0.1)
    result = cbf.filter(
        nominal_control=nominal,
        ego=vehicle("ego", 0.0, 0.0, 0.3, heading=0.06),
        obstacles=[],
        road=RoadGeometry(heading=0.0, left_clearance=0.9, right_clearance=3.0),
    )

    assert result.fallback_mode != "low_speed_road_recovery"
    assert result.intervened


def test_low_speed_recovery_requires_a_meaningful_boundary_heading_component():
    cbf = PredictiveEscapeCBF(
        CBFConfig(
            passthrough_without_active_obstacle=False,
            road_low_speed_passthrough=2.0,
            road_low_speed_min_outward_heading_component=0.03,
        )
    )
    result = cbf.filter(
        nominal_control=Control2D(3.0, 0.1),
        ego=vehicle("ego", 0.0, 0.0, 0.3, heading=0.0),
        obstacles=[],
        road=RoadGeometry(heading=0.0, left_clearance=1.0, right_clearance=3.0),
    )

    assert result.fallback_mode != "low_speed_road_recovery"
    assert result.intervened


def test_low_speed_recovery_accepts_heading_away_from_the_tight_boundary():
    cbf = PredictiveEscapeCBF(
        CBFConfig(
            passthrough_without_active_obstacle=False,
            road_low_speed_passthrough=2.0,
            road_low_speed_min_outward_heading_component=0.03,
        )
    )
    result = cbf.filter(
        nominal_control=Control2D(3.0, -0.1),
        ego=vehicle("ego", 0.0, 0.0, 0.3, heading=-0.06),
        obstacles=[],
        road=RoadGeometry(heading=0.0, left_clearance=1.0, right_clearance=3.0),
    )

    assert result.fallback_mode == "low_speed_road_recovery"
    assert not result.intervened



def test_front_rear_squeeze_generates_a_lateral_escape():
    cbf = PredictiveEscapeCBF(CBFConfig(other_accel_uncertainty=0.0))
    result = cbf.filter(
        nominal_control=Control2D(0.0, 0.0),
        ego=vehicle("ego", 0.0, 0.0, 10.0),
        obstacles=[
            vehicle("rear", -10.0, 0.0, 15.0),
            vehicle("front", 10.0, 0.0, 8.0),
        ],
        road=RoadGeometry(heading=0.0, left_clearance=5.0, right_clearance=5.0),
    )

    assert result.mode in {"left_escape", "right_escape"}
    assert abs(result.control.curvature) > 1e-3
    assert result.candidate_scores[result.mode] > result.candidate_scores["nominal"]


def test_left_side_vehicle_makes_the_filter_choose_right():
    cbf = PredictiveEscapeCBF(
        CBFConfig(
            other_accel_uncertainty=0.0,
            activation_require_current_lateral_overlap=False,
        )
    )
    result = cbf.filter(
        nominal_control=Control2D(0.0, 0.0),
        ego=vehicle("ego", 0.0, 0.0, 10.0),
        obstacles=[
            vehicle("rear", -10.0, 0.0, 15.0),
            vehicle("left", 0.0, 3.5, 10.0),
        ],
        road=RoadGeometry(heading=0.0, left_clearance=6.0, right_clearance=6.0),
    )

    assert result.mode == "right_escape"
    assert result.control.curvature < 0.0


def test_metadrive_adapter_is_external_and_returns_normalized_action():
    ego = _FakeVehicle("ego", [0.0, 0.0], [10.0, 0.0])
    rear = _FakeVehicle("rear", [-10.0, 0.0], [15.0, 0.0])
    env = SimpleNamespace(
        agent=ego,
        engine=SimpleNamespace(traffic_manager=SimpleNamespace(vehicles=[ego, rear])),
    )
    plugin = AdaptiveMetaDriveCBFPlugin(CBFConfig(other_accel_uncertainty=0.0))
    plugin.reset(env)

    safe_action, telemetry = plugin.filter_action(env, np.asarray([0.0, 0.0]))

    assert safe_action.shape == (2,)
    assert np.all(safe_action >= -1.0)
    assert np.all(safe_action <= 1.0)
    assert telemetry["rta_enabled"]
    assert telemetry["rta_plugin"] == "predictive_escape_cbf2d"
    assert telemetry["rta_obstacle_count"] == 1
    assert telemetry["rta_active_obstacle_count"] == 1

def test_metadrive_adapter_uses_side_detector_road_boundaries():
    ego = _FakeVehicle("ego", [0.0, 0.0], [0.0, 0.0])
    ego.config = {
        "side_detector": {
            "num_lasers": 4,
            "distance": 50.0,
        }
    }
    ego.side_detector = _FakeSideDetector([0.04, 1.0, 0.06, 1.0])
    ego.engine = SimpleNamespace(
        physics_world=SimpleNamespace(static_world=object())
    )
    plugin = AdaptiveMetaDriveCBFPlugin(
        CBFConfig(road_use_side_detector=True)
    )

    road = plugin._road_geometry(ego)

    assert plugin._road_geometry_source == "side_detector"
    np.testing.assert_allclose(road.left_clearance, 2.0, atol=1e-9)
    np.testing.assert_allclose(road.right_clearance, 3.0, atol=1e-9)


class _FakeSideDetector:
    start_phase_offset = np.pi / 2.0

    def __init__(self, cloud_points):
        self.cloud_points = cloud_points

    def perceive(self, *args, **kwargs):
        return SimpleNamespace(cloud_points=self.cloud_points)

class _FakeVehicle:
    LENGTH = 4.5
    WIDTH = 2.0

    def __init__(self, name, position, velocity):
        self.name = name
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.heading_theta = 0.0
        self.max_steering = 40.0
        self.dist_to_left_side = 5.0
        self.dist_to_right_side = 5.0
        self.navigation = SimpleNamespace(current_heading_theta_at_long=0.0)
