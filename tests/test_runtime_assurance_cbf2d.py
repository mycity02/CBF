from types import SimpleNamespace

import numpy as np
import pytest

from cbf_rta.adaptive_plugin import (
    AdaptiveMetaDriveCBFPlugin,
    FSMHOCBFPlugin,
    PredictiveEscapeCBF,
)
from cbf_rta.fsm import (
    FSMHOCBF,
    FSMState,
    SafetySupervisorFSM,
    ThreatAssessment,
)
from cbf_rta.metrics import termination_outcome, termination_summary
from cbf_rta.qp import (
    LinearConstraint,
    solve_prioritized_projection,
    solve_weighted_projection,
)
from cbf_rta.types import (
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
        road=RoadGeometry(heading=0.0, left_clearance=1.15, right_clearance=3.0),
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
        road=RoadGeometry(heading=0.0, left_clearance=1.15, right_clearance=3.0),
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
    ego.engine = SimpleNamespace(physics_world=SimpleNamespace(static_world=object()))
    plugin = AdaptiveMetaDriveCBFPlugin(CBFConfig(road_use_side_detector=True))

    road = plugin._road_geometry(ego)

    assert plugin._road_geometry_source == "reference_route+side_detector"
    np.testing.assert_allclose(road.left_clearance, 2.0, atol=1e-9)
    np.testing.assert_allclose(road.right_clearance, 3.0, atol=1e-9)


def test_metadrive_adapter_side_detector_never_relaxes_reference_route():
    ego = _FakeVehicle("ego", [0.0, 0.0], [0.0, 0.0])
    ego.dist_to_left_side = 1.0
    ego.dist_to_right_side = 3.0
    ego.config = {
        "side_detector": {
            "num_lasers": 4,
            "distance": 50.0,
        }
    }
    ego.side_detector = _FakeSideDetector([0.20, 1.0, 0.20, 1.0])
    ego.engine = SimpleNamespace(physics_world=SimpleNamespace(static_world=object()))
    plugin = AdaptiveMetaDriveCBFPlugin(CBFConfig(road_use_side_detector=True))

    road = plugin._road_geometry(ego)

    np.testing.assert_allclose(road.left_clearance, 1.0, atol=1e-9)
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


def test_fsm_squeeze_selects_one_persistent_mode_and_solves_one_qp():
    cbf = FSMHOCBF(
        CBFConfig(
            other_accel_uncertainty=0.0,
            passthrough_without_active_obstacle=True,
        )
    )
    result = cbf.filter(
        nominal_control=Control2D(0.0, 0.0),
        ego=vehicle("ego", 0.0, 0.0, 10.0),
        obstacles=[
            vehicle("rear", -10.0, 0.0, 15.0),
            vehicle("front", 10.0, 0.0, 8.0),
        ],
        road=RoadGeometry(heading=0.0, left_clearance=5.0, right_clearance=5.0),
    )

    assert result.mode in {"escape_left", "escape_right"}
    assert result.qp_solve_count == 1
    assert result.candidate_scores == {}
    assert result.reference_control is not None
    assert abs(result.reference_control.curvature) > 1e-3


def test_fsm_uses_side_margin_to_select_the_open_escape_corridor():
    cbf = FSMHOCBF(CBFConfig(other_accel_uncertainty=0.0))
    result = cbf.filter(
        nominal_control=Control2D(0.0, 0.0),
        ego=vehicle("ego", 0.0, 0.0, 10.0),
        obstacles=[
            vehicle("rear", -10.0, 0.0, 15.0),
            vehicle("left", 0.0, 3.5, 10.0),
        ],
        road=RoadGeometry(heading=0.0, left_clearance=6.0, right_clearance=6.0),
    )

    assert result.mode == "escape_right"
    assert result.reference_control.curvature < 0.0
    assert result.fsm_right_margin > result.fsm_left_margin


def test_fsm_dwell_prevents_left_right_chattering():
    supervisor = SafetySupervisorFSM(
        CBFConfig(fsm_min_dwell_steps=3, fsm_direction_hysteresis=0.0)
    )
    left_preferred = ThreatAssessment(
        front_threat=False,
        rear_threat=True,
        front_clear=True,
        rear_clear=False,
        left_margin=1.0,
        right_margin=0.2,
    )
    right_preferred = ThreatAssessment(
        front_threat=False,
        rear_threat=True,
        front_clear=True,
        rear_clear=False,
        left_margin=0.2,
        right_margin=1.0,
    )

    first = supervisor.update(left_preferred, nominal_curvature=0.0)
    second = supervisor.update(right_preferred, nominal_curvature=0.0)

    assert first.state is FSMState.ESCAPE_LEFT
    assert second.state is FSMState.ESCAPE_LEFT
    assert not second.transitioned


def test_fsm_clear_hysteresis_passes_through_recovery_before_nominal():
    supervisor = SafetySupervisorFSM(
        CBFConfig(
            fsm_min_dwell_steps=0,
            fsm_clear_steps=2,
            fsm_recovery_steps=2,
        )
    )
    threat = ThreatAssessment(False, True, True, False, 1.0, 1.0)
    clear = ThreatAssessment(False, False, True, True, 1.0, 1.0)

    assert supervisor.update(threat, 0.0).state is FSMState.ESCAPE_LEFT
    assert supervisor.update(clear, 0.0).state is FSMState.ESCAPE_LEFT
    assert supervisor.update(clear, 0.0).state is FSMState.RECOVERY
    assert supervisor.update(clear, 0.0).state is FSMState.RECOVERY
    assert supervisor.update(clear, 0.0).state is FSMState.NOMINAL


def test_fsm_metadrive_plugin_reports_state_and_single_qp():
    ego = _FakeVehicle("ego", [0.0, 0.0], [10.0, 0.0])
    rear = _FakeVehicle("rear", [-10.0, 0.0], [15.0, 0.0])
    env = SimpleNamespace(
        agent=ego,
        engine=SimpleNamespace(traffic_manager=SimpleNamespace(vehicles=[ego, rear])),
    )
    plugin = FSMHOCBFPlugin(CBFConfig(other_accel_uncertainty=0.0))
    plugin.reset(env)

    safe_action, telemetry = plugin.filter_action(env, np.asarray([0.0, 0.0]))

    assert safe_action.shape == (2,)
    assert telemetry["rta_plugin"] == "fsm_hocbf_qp"
    assert telemetry["rta_mode"] in {"escape_left", "escape_right"}
    assert telemetry["rta_fsm_transitioned"]
    assert telemetry["rta_qp_solve_count"] == 1
    assert telemetry["rta_candidate_scores"] == {}


def test_fsm_nominal_passthrough_still_reports_current_road_barriers():
    cbf = FSMHOCBF(
        CBFConfig(
            passthrough_without_active_obstacle=True,
            road_low_speed_passthrough=0.0,
        )
    )

    result = cbf.filter(
        nominal_control=Control2D(0.0, 0.0),
        ego=vehicle("ego", 0.0, 0.0, 10.0),
        obstacles=[],
        road=RoadGeometry(heading=0.0, left_clearance=5.0, right_clearance=5.0),
    )

    assert result.qp_solve_count == 0
    assert not result.intervened
    assert len([item for item in result.diagnostics if item.kind == "road"]) == 2
    assert result.to_telemetry()["rta_min_road_h"] is not None


def test_fsm_far_from_road_boundary_does_not_filter_an_outward_nominal_action():
    cbf = FSMHOCBF(
        CBFConfig(
            passthrough_without_active_obstacle=True,
            road_low_speed_passthrough=0.0,
            road_activation_h=0.5,
            road_activation_psi1=0.0,
        )
    )
    nominal = Control2D(0.0, 0.3)

    result = cbf.filter(
        nominal_control=nominal,
        ego=vehicle("ego", 0.0, 0.0, 10.0),
        obstacles=[],
        road=RoadGeometry(heading=0.0, left_clearance=5.0, right_clearance=5.0),
    )

    assert result.qp_solve_count == 0
    np.testing.assert_allclose(result.control.as_array(), nominal.as_array())


def test_fsm_nominal_passthrough_cannot_bypass_an_unsafe_road_constraint():
    cbf = FSMHOCBF(
        CBFConfig(
            passthrough_without_active_obstacle=True,
            road_low_speed_passthrough=0.0,
        )
    )

    result = cbf.filter(
        nominal_control=Control2D(0.0, 0.0),
        ego=vehicle("ego", 0.0, 0.0, 2.0, heading=0.05),
        obstacles=[],
        road=RoadGeometry(heading=0.0, left_clearance=1.35, right_clearance=5.0),
    )

    assert result.qp_solve_count == 1
    assert result.intervened
    assert result.control.curvature < 0.0
    left = min(
        (item for item in result.diagnostics if item.name.startswith("road:left")),
        key=lambda item: item.h,
    )
    assert left.h > 0.0
    assert result.fallback_mode != "low_speed_road_recovery"
    assert left.residual >= -cbf.config.constraint_tolerance


def test_fsm_low_speed_negative_road_h_uses_directed_recovery():
    config = CBFConfig(
        road_margin=0.20,
        passthrough_without_active_obstacle=True,
        road_low_speed_passthrough=2.0,
        fsm_road_recovery_max_initial_penetration=0.25,
        fsm_road_recovery_acceleration=1.0,
    )
    cbf = FSMHOCBF(config)
    ego = vehicle("ego", 0.0, 0.0, 0.0, heading=0.06)
    road = RoadGeometry(heading=0.0, left_clearance=1.15, right_clearance=3.0)
    nominal = Control2D(3.0, 0.1)
    left = min(
        (
            item
            for item in cbf._road_barriers(ego, road)
            if item.diagnostic.name.startswith("road:left")
        ),
        key=lambda item: item.diagnostic.h,
    )

    assert -config.fsm_road_recovery_max_initial_penetration < left.diagnostic.h < 0.0
    result = cbf.filter(nominal, ego, [], road)

    assert result.fallback_mode == "low_speed_road_recovery"
    assert np.isclose(
        result.control.acceleration,
        config.fsm_road_recovery_acceleration,
    )
    assert result.control.curvature < 0.0
    assert result.intervened


def test_fsm_road_recovery_rejects_excessive_initial_penetration():
    cbf = FSMHOCBF(
        CBFConfig(
            road_margin=0.20,
            passthrough_without_active_obstacle=True,
            road_low_speed_passthrough=2.0,
            fsm_road_recovery_max_initial_penetration=0.10,
        )
    )
    result = cbf.filter(
        Control2D(3.0, 0.1),
        vehicle("ego", 0.0, 0.0, 0.0, heading=0.06),
        [],
        RoadGeometry(heading=0.0, left_clearance=1.15, right_clearance=3.0),
    )

    assert result.fallback_mode == "road_recovery_fail_stop"
    assert result.qp_solve_count == 0
    assert not result.hard_feasible
    assert not result.mandatory_feasible
    assert result.control.acceleration <= 0.0
    assert result.control.curvature < 0.0
    assert result.max_constraint_violation > 0.0
    assert result.normalized_constraint_violation > 0.0


@pytest.mark.parametrize(
    "heading,left_clearance,right_clearance,nominal_curvature,expected_sign",
    [
        (0.06, 1.15, 3.0, 0.1, -1.0),
        (-0.06, 3.0, 1.15, -0.1, 1.0),
    ],
)
def test_fsm_low_speed_recovery_is_symmetric_at_both_road_boundaries(
    heading,
    left_clearance,
    right_clearance,
    nominal_curvature,
    expected_sign,
):
    cbf = FSMHOCBF(
        CBFConfig(
            road_margin=0.20,
            passthrough_without_active_obstacle=True,
            road_low_speed_passthrough=2.0,
        )
    )

    result = cbf.filter(
        Control2D(3.0, nominal_curvature),
        vehicle("ego", 0.0, 0.0, 0.0, heading=heading),
        [],
        RoadGeometry(
            heading=0.0,
            left_clearance=left_clearance,
            right_clearance=right_clearance,
        ),
    )

    assert result.fallback_mode == "low_speed_road_recovery"
    assert result.hard_feasible
    assert result.mandatory_feasible
    assert result.control.acceleration > 0.0
    assert expected_sign * result.control.curvature > 0.0
    assert all(
        item.residual >= -cbf.config.constraint_tolerance
        for item in result.diagnostics
        if item.kind == "road"
    )


def test_fsm_raw_corner_cbf_recovery_keeps_nominal_acceleration_when_feasible():
    heading = 0.059
    speed = 0.333
    ego = VehicleState2D(
        name="ego",
        position=np.zeros(2),
        velocity=speed * np.asarray([np.cos(heading), np.sin(heading)]),
        acceleration=np.zeros(2),
        heading=heading,
        length=4.515,
        width=1.852,
        kind="vehicle",
    )
    projected_extent = 0.5 * ego.length * np.sin(heading) + 0.5 * ego.width * np.cos(
        heading
    )
    cbf = FSMHOCBF(
        CBFConfig(
            road_margin=0.05,
            passthrough_without_active_obstacle=True,
            road_low_speed_passthrough=2.0,
        )
    )

    result = cbf.filter(
        Control2D(3.0, 0.134),
        ego,
        [],
        RoadGeometry(
            heading=0.0,
            left_clearance=projected_extent + 0.05 - 0.10859,
            right_clearance=5.0,
        ),
    )

    left_front = next(
        item for item in result.diagnostics if item.name == "road:left:front_left"
    )
    assert np.isclose(left_front.h, -0.10859)
    assert result.fallback_mode == "directed_road_recovery"
    assert result.hard_feasible
    assert np.isclose(result.control.acceleration, 3.0)
    assert result.control.curvature < 0.0
    assert left_front.residual >= -cbf.config.constraint_tolerance
    assert np.isclose(left_front.psi1, left_front.residual)


def test_fsm_low_speed_road_recovery_and_rear_constraint_share_one_qp():
    cbf = FSMHOCBF(
        CBFConfig(
            road_margin=0.20,
            passthrough_without_active_obstacle=True,
            road_low_speed_passthrough=2.0,
            other_accel_uncertainty=0.0,
        )
    )
    result = cbf.filter(
        Control2D(3.0, 0.1),
        vehicle("ego", 0.0, 0.0, 0.0, heading=0.06),
        [vehicle("rear", -5.0, 0.0, 0.1)],
        RoadGeometry(heading=0.0, left_clearance=1.15, right_clearance=3.0),
    )

    assert result.fallback_mode == "low_speed_road_recovery"
    assert result.active_obstacle_count == 1
    assert result.qp_solve_count == 1
    assert result.hard_feasible
    assert result.mandatory_feasible
    assert result.control.acceleration > 0.0
    assert result.control.curvature < 0.0
    assert any(item.kind == "vehicle" for item in result.diagnostics)
    assert all(
        item.residual >= -cbf.config.constraint_tolerance for item in result.diagnostics
    )
    recovered = next(
        item
        for item in result.diagnostics
        if item.kind == "road" and item.h < 0.0
    )
    assert np.isclose(recovered.nominal_h, recovered.h)
    assert recovered.effective_h > 0.0
    assert np.isclose(recovered.effective_residual, recovered.residual)
    assert np.isclose(
        recovered.psi1,
        recovered.h_dot + cbf.config.alpha1 * recovered.nominal_h,
    )
    assert recovered.psi1 < 0.0
    assert recovered.effective_residual >= -cbf.config.constraint_tolerance


def test_fsm_unrecoverable_road_set_does_not_fail_stop_into_a_rear_threat():
    cbf = FSMHOCBF(
        CBFConfig(
            road_margin=0.20,
            passthrough_without_active_obstacle=True,
            road_low_speed_passthrough=2.0,
            fsm_road_recovery_max_initial_penetration=0.10,
            other_accel_uncertainty=0.0,
        )
    )
    result = cbf.filter(
        Control2D(3.0, 0.1),
        vehicle("ego", 0.0, 0.0, 0.0, heading=0.06),
        [vehicle("rear", -5.0, 0.0, 0.1)],
        RoadGeometry(heading=0.0, left_clearance=1.15, right_clearance=3.0),
    )

    assert result.active_obstacle_count == 1
    assert result.qp_solve_count == 1
    assert result.fallback_mode == "road_recovery_conflict+relaxed_road"
    assert not result.hard_feasible
    assert result.mandatory_feasible
    assert result.control.acceleration > 0.0
    vehicle_diagnostic = next(
        item for item in result.diagnostics if item.kind == "vehicle"
    )
    assert vehicle_diagnostic.residual >= -cbf.config.constraint_tolerance


def test_fsm_nominal_safe_front_vehicle_does_not_disable_road_fail_stop():
    cbf = FSMHOCBF(
        CBFConfig(
            road_margin=0.20,
            passthrough_without_active_obstacle=True,
            road_low_speed_passthrough=2.0,
            fsm_road_recovery_max_initial_penetration=0.10,
            other_accel_uncertainty=0.0,
        )
    )
    result = cbf.filter(
        Control2D(3.0, 0.1),
        vehicle("ego", 0.0, 0.0, 0.0, heading=0.06),
        [vehicle("front", 10.0, 0.0, 0.0)],
        RoadGeometry(heading=0.0, left_clearance=1.15, right_clearance=3.0),
    )

    assert result.active_obstacle_count == 1
    assert result.fallback_mode == "road_recovery_fail_stop"
    assert result.qp_solve_count == 0
    assert result.control.acceleration <= 0.0
    vehicle_diagnostic = next(
        item for item in result.diagnostics if item.kind == "vehicle"
    )
    assert vehicle_diagnostic.residual >= -cbf.config.constraint_tolerance


def test_fsm_road_recovery_latches_until_release_and_reset_clears_it():
    config = CBFConfig(
        road_margin=0.20,
        passthrough_without_active_obstacle=True,
        road_low_speed_passthrough=2.0,
    )
    cbf = FSMHOCBF(config)
    road_tight = RoadGeometry(
        heading=0.0,
        left_clearance=1.15,
        right_clearance=3.0,
    )

    entering = cbf.filter(
        Control2D(3.0, 0.1),
        vehicle("ego", 0.0, 0.0, 0.0, heading=0.06),
        [],
        road_tight,
    )
    latched = cbf.filter(
        Control2D(3.0, 0.1),
        vehicle("ego", 0.0, 0.0, 2.5, heading=0.06),
        [],
        RoadGeometry(heading=0.0, left_clearance=1.20, right_clearance=3.0),
    )
    released = cbf.filter(
        Control2D(3.0, -0.03),
        vehicle("ego", 0.0, 0.0, 2.5, heading=0.06),
        [],
        RoadGeometry(heading=0.0, left_clearance=1.36, right_clearance=3.0),
    )

    assert entering.fallback_mode == "low_speed_road_recovery"
    assert latched.fallback_mode == "low_speed_road_recovery"
    assert released.fallback_mode == "inactive"
    assert cbf._road_recovery_barrier_name is None

    cbf.filter(
        Control2D(3.0, 0.1),
        vehicle("ego", 0.0, 0.0, 0.0, heading=0.06),
        [],
        road_tight,
    )
    assert cbf._road_recovery_barrier_name == "road:left"
    cbf.reset()
    after_reset = cbf.filter(
        Control2D(3.0, -0.03),
        vehicle("ego", 0.0, 0.0, 2.5, heading=0.06),
        [],
        RoadGeometry(heading=0.0, left_clearance=1.20, right_clearance=3.0),
    )
    assert cbf._road_recovery_barrier_name is None
    assert "road_recovery" not in after_reset.fallback_mode


def test_road_barrier_uses_heading_dependent_vehicle_footprint():
    cbf = FSMHOCBF(CBFConfig(road_margin=0.05))
    ego = vehicle("ego", 0.0, 0.0, 10.0, heading=np.pi / 6.0)

    barriers = cbf._road_barriers(
        ego,
        RoadGeometry(heading=0.0, left_clearance=2.0, right_clearance=5.0),
    )

    left = min(
        (item for item in barriers if item.diagnostic.name.startswith("road:left")),
        key=lambda item: item.diagnostic.h,
    )
    expected_extent = 0.5 * ego.length * np.sin(np.pi / 6.0) + 0.5 * ego.width * np.cos(
        np.pi / 6.0
    )
    assert np.isclose(left.diagnostic.h, 2.0 - expected_extent - 0.05)
    assert left.diagnostic.h < 0.0


def test_fsm_corner_road_cbf_steers_inward_without_braking_at_low_speed():
    heading = 0.059
    ego = VehicleState2D(
        name="ego",
        position=np.zeros(2),
        velocity=0.333 * np.asarray([np.cos(heading), np.sin(heading)]),
        acceleration=np.zeros(2),
        heading=heading,
        length=4.515,
        width=1.852,
        kind="vehicle",
    )
    projected_extent = 0.5 * ego.length * np.sin(heading) + 0.5 * ego.width * np.cos(
        heading
    )
    left_clearance = projected_extent + 0.05 - 0.10859
    cbf = FSMHOCBF(
        CBFConfig(
            road_margin=0.05,
            passthrough_without_active_obstacle=True,
            road_low_speed_passthrough=0.0,
        )
    )

    result = cbf.filter(
        Control2D(3.0, 0.134),
        ego,
        [],
        RoadGeometry(heading=0.0, left_clearance=left_clearance, right_clearance=5.0),
    )

    left_front = next(
        item for item in result.diagnostics if item.name == "road:left:front_left"
    )
    assert np.isclose(left_front.h, -0.10859)
    assert result.hard_feasible
    assert result.fallback_mode == "none"
    assert np.isclose(result.control.acceleration, 3.0)
    assert result.control.curvature <= -0.212
    assert left_front.residual >= -cbf.config.constraint_tolerance


def test_fsm_rear_threat_does_not_wait_for_full_lateral_overlap():
    cbf = FSMHOCBF(CBFConfig(other_accel_uncertainty=0.0))
    road = RoadGeometry(heading=0.0, left_clearance=6.0, right_clearance=6.0)
    ego = vehicle("ego", 0.0, 0.0, 10.0)
    offset_rear = vehicle("offset_rear", -2.0, 2.6, 12.0)

    assert not cbf._current_lateral_overlap(ego, offset_rear, road)
    result = cbf.filter(
        nominal_control=Control2D(0.0, 0.0),
        ego=ego,
        obstacles=[offset_rear],
        road=road,
    )

    assert result.mode == "escape_right"
    assert result.transition_reason.startswith("rear_threat")
    assert result.active_obstacle_count == 1
    assert result.qp_solve_count == 1
    assert result.reference_control.curvature < 0.0


def test_fsm_non_overlapping_side_vehicle_stays_out_of_hard_qp():
    cbf = FSMHOCBF(
        CBFConfig(
            other_accel_uncertainty=0.0,
            passthrough_without_active_obstacle=True,
            activation_require_current_lateral_overlap=True,
        )
    )
    road = RoadGeometry(heading=0.0, left_clearance=6.0, right_clearance=6.0)
    ego = vehicle("ego", 0.0, 0.0, 10.0)
    side = vehicle("side", 15.0, 3.0, 0.0)
    nominal = Control2D(3.0, 0.0)

    assert not cbf._current_lateral_overlap(ego, side, road)
    result = cbf.filter(nominal, ego, [side], road)

    assert result.active_obstacle_count == 0
    assert result.qp_solve_count == 0
    assert not result.intervened
    np.testing.assert_allclose(result.control.as_array(), nominal.as_array())


def test_fsm_non_overlapping_vulnerable_road_user_remains_active():
    cbf = FSMHOCBF(CBFConfig(other_accel_uncertainty=0.0))
    road = RoadGeometry(heading=0.0, left_clearance=6.0, right_clearance=6.0)
    ego = vehicle("ego", 0.0, 0.0, 10.0)
    pedestrian = vehicle("pedestrian", 15.0, 3.0, 0.0, kind="pedestrian")

    assert not cbf._current_lateral_overlap(ego, pedestrian, road)
    assert cbf._current_active_obstacles(ego, [pedestrian], road) == [pedestrian]


def test_fsm_non_overlapping_front_cutin_remains_active():
    cbf = FSMHOCBF(CBFConfig(other_accel_uncertainty=0.0))
    road = RoadGeometry(heading=0.0, left_clearance=6.0, right_clearance=6.0)
    ego = vehicle("ego", 0.0, 0.0, 10.0)
    cutin = vehicle("cutin", 10.0, 3.0, 10.0, heading=-0.20)

    assert not cbf._current_lateral_overlap(ego, cutin, road)
    active = cbf._current_active_obstacles(ego, [cutin], road)

    assert active == [cutin]


def test_fsm_far_offset_rear_vehicle_stays_inactive():
    cbf = FSMHOCBF(CBFConfig(other_accel_uncertainty=0.0))
    road = RoadGeometry(heading=0.0, left_clearance=6.0, right_clearance=6.0)
    ego = vehicle("ego", 0.0, 0.0, 10.0)
    offset_rear = vehicle("offset_rear", -2.0, 3.5, 12.0)

    assert not cbf._current_lateral_overlap(ego, offset_rear, road)
    assert cbf._current_active_obstacles(ego, [offset_rear], road) == []


def test_fsm_nominal_safe_front_constraint_does_not_enter_brake():
    cbf = FSMHOCBF(CBFConfig(other_accel_uncertainty=0.0))
    nominal = Control2D(3.0, 0.0)
    result = cbf.filter(
        nominal,
        vehicle("ego", 0.0, 0.0, 10.0),
        [vehicle("front", 10.0, 0.0, 10.0)],
        RoadGeometry(heading=0.0, left_clearance=6.0, right_clearance=6.0),
    )

    assert result.active_obstacle_count == 1
    assert result.qp_solve_count == 1
    assert result.mode == "nominal"
    assert not result.intervened
    np.testing.assert_allclose(result.control.as_array(), nominal.as_array())


def test_fsm_nominal_unsafe_front_constraint_enters_brake():
    cbf = FSMHOCBF(CBFConfig(other_accel_uncertainty=0.0))
    nominal = Control2D(3.0, 0.0)
    ego = vehicle("ego", 0.0, 0.0, 10.0)
    front = vehicle("front", 10.0, 0.0, 8.0)
    road = RoadGeometry(heading=0.0, left_clearance=6.0, right_clearance=6.0)
    barrier = cbf._vehicle_barrier(ego, front, road)

    assert barrier.diagnostic.h > 0.0
    assert barrier.constraint.residual(nominal.as_array()) < 0.0
    result = cbf.filter(nominal, ego, [front], road)

    assert result.mode == "brake"
    assert result.transition_reason.startswith("front_threat")
    assert result.intervened


def test_fsm_unsafe_front_h_enters_brake_even_if_nominal_residual_is_safe():
    cbf = FSMHOCBF(CBFConfig(other_accel_uncertainty=0.0))
    nominal = Control2D(3.0, 0.0)
    ego = vehicle("ego", 0.0, 0.0, 10.0)
    front = vehicle("front", 6.0, 0.0, 15.0)
    road = RoadGeometry(heading=0.0, left_clearance=6.0, right_clearance=6.0)
    barrier = cbf._vehicle_barrier(ego, front, road)

    assert barrier.diagnostic.h < 0.0
    assert barrier.constraint.residual(nominal.as_array()) > 0.0
    result = cbf.filter(nominal, ego, [front], road)

    assert result.mode == "brake"
    assert result.transition_reason.startswith("front_threat")
    assert result.intervened


def test_fsm_escape_reference_is_current_state_feedback_without_a_horizon():
    cbf = FSMHOCBF(CBFConfig())
    road = RoadGeometry(heading=0.0, left_clearance=5.0, right_clearance=5.0)

    entering = cbf._reference_for_state(
        FSMState.ESCAPE_LEFT,
        Control2D(0.0, 0.0),
        vehicle("ego", 0.0, 0.0, 10.0, heading=0.0),
        road,
    )
    correcting = cbf._reference_for_state(
        FSMState.ESCAPE_LEFT,
        Control2D(0.0, 0.0),
        vehicle("ego", 0.0, 0.0, 10.0, heading=0.3),
        road,
    )

    assert entering.curvature > 0.0
    assert correcting.curvature < 0.0
    assert not hasattr(cbf.config, "fsm_maneuver_horizon")


def test_termination_summary_separates_events_from_exclusive_outcomes():
    episodes = [
        {
            "termination_reason": termination_outcome(True, True, False),
            "out_of_road": True,
            "degenerate_scenario": False,
        },
        {
            "termination_reason": termination_outcome(False, True, False),
            "out_of_road": True,
            "degenerate_scenario": False,
        },
        {
            "termination_reason": termination_outcome(False, False, True),
            "out_of_road": False,
            "degenerate_scenario": False,
        },
        {
            "termination_reason": termination_outcome(False, False, False),
            "out_of_road": False,
            "degenerate_scenario": False,
        },
        {
            "termination_reason": termination_outcome(True, True, False),
            "out_of_road": True,
            "degenerate_scenario": True,
        },
    ]

    for index, episode in enumerate(episodes):
        episode["reward"] = float(index)
        episode["route_completion"] = 0.1 * index
        episode["collision"] = index == 2
        episode["arrive_dest"] = index in (0, 4)

    summary = termination_summary(episodes)

    assert summary["termination_counts"] == {
        "success": 2,
        "road_violation": 1,
        "collision": 1,
        "timeout_or_other": 1,
    }
    assert summary["degenerate_scenario_count"] == 1
    assert summary["valid_dynamic_out_of_road_rate"] == 0.5
    assert summary["valid_dynamic_road_violation_termination_rate"] == 0.25
    assert summary["valid_dynamic_reward"] == 1.5
    assert np.isclose(summary["valid_dynamic_completion"], 0.15)
    assert summary["valid_dynamic_collision_rate"] == 0.25
    assert summary["valid_dynamic_arrive_dest_rate"] == 0.25


def test_fsm_front_uses_open_escape_corridor_before_braking():
    supervisor = SafetySupervisorFSM(
        CBFConfig(
            fsm_min_side_margin=0.0,
            fsm_front_escape_enabled=True,
            fsm_front_escape_min_margin=0.2,
        )
    )
    front_only = ThreatAssessment(True, False, False, True, 1.0, 0.5)

    decision = supervisor.update(front_only, nominal_curvature=0.0)

    assert decision.state is FSMState.ESCAPE_LEFT
    assert decision.reason == "front_threat_left_margin"


def test_fsm_front_brakes_when_no_safe_corridor_then_escapes_from_rear_threat():
    supervisor = SafetySupervisorFSM(
        CBFConfig(
            fsm_min_side_margin=0.0,
            fsm_front_escape_enabled=True,
            fsm_front_escape_min_margin=0.2,
        )
    )
    front_only = ThreatAssessment(True, False, False, True, 0.1, 0.1)
    squeeze = ThreatAssessment(True, True, False, False, 1.0, 0.5)

    braking = supervisor.update(front_only, nominal_curvature=0.0)
    escaping = supervisor.update(squeeze, nominal_curvature=0.0)

    assert braking.state is FSMState.BRAKE
    assert escaping.state is FSMState.ESCAPE_LEFT


def test_fsm_adaptive_brake_scales_with_current_front_risk_and_releases():
    cbf = FSMHOCBF(
        CBFConfig(
            fsm_enter_h=0.8,
            fsm_enter_psi1=0.5,
            fsm_brake_acceleration=-2.0,
        )
    )
    mild = ThreatAssessment(
        True, False, False, True, 0.0, 0.0, front_h=0.6, front_psi1=0.5
    )
    severe = ThreatAssessment(
        True, False, False, True, 0.0, 0.0, front_h=0.0, front_psi1=0.5
    )
    hysteresis_only = ThreatAssessment(
        False, False, False, True, 0.0, 0.0, front_h=0.9, front_psi1=0.7
    )

    assert np.isclose(cbf._adaptive_brake_acceleration(1.0, mild), -0.5)
    assert np.isclose(cbf._adaptive_brake_acceleration(1.0, severe), -2.0)
    assert np.isclose(cbf._adaptive_brake_acceleration(1.0, hysteresis_only), 1.0)

    threshold = ThreatAssessment(
        True, False, False, True, 0.0, 0.0, front_h=0.8, front_psi1=0.5
    )
    cbf.supervisor.state_age_steps = 30
    assert np.isclose(cbf._adaptive_brake_acceleration(1.0, threshold), -2.0)
