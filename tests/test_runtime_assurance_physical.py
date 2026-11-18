from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cbf_rta.adaptive_plugin import RearEscapeCBFPlugin
from cbf_rta.fsm import FSMState
from cbf_rta.metadrive_plugin import MetaDriveCBFPlugin
from cbf_rta.qp import LinearConstraint, solve_prioritized_projection
from cbf_rta.rear_escape import ROAD_MODEL, RearEscapeCBF
from cbf_rta.rear_guard import RearPhysicalGuardPlugin
from cbf_rta.types import CBFConfig, Control2D, RoadGeometry, VehicleState2D


def native_car(speed=10.):
    return SimpleNamespace(name='ego', MASS=1100., LENGTH=4.515, WIDTH=1.852,
        FRONT_WHEELBASE=1.05234, REAR_WHEELBASE=1.4166, max_steering=40.,
        position=np.zeros(2), velocity=np.array([speed, 0.]), heading_theta=0.,
        dist_to_left_side=5., dist_to_right_side=5., metadrive_type='vehicle',
        config={'max_engine_force':759.463623046875, 'max_brake_force':89.463623046875},
        engine=SimpleNamespace(global_config={'physics_world_step_size':.02}))


def physical_config(**values):
    params = dict(rear_use_native_dynamics=True, rear_use_acceleration_metric=True,
                  rear_velocity_road_barriers=True, qp_preconditioned_fallback=True,
                  rear_lateral_acceleration_weight=4., road_margin=.05,
                  passthrough_without_active_obstacle=True)
    params.update(values)
    return CBFConfig(**params)


def test_native_force_impulse_mapping_and_inverse_are_consistent():
    ego = native_car()
    plugin = RearEscapeCBFPlugin(physical_config())
    old = MetaDriveCBFPlugin(CBFConfig())
    for throttle in [-.5, -.1, 0., .1, 1.]:
        raw = np.array([.08, throttle])
        control, limit = plugin._action_to_control(raw, ego)
        action = plugin._control_to_action(control, ego, limit)
        np.testing.assert_allclose(action, raw, atol=1e-7)
    assert old._action_to_control(np.array([0., -.5]), ego)[0].acceleration == -3.
    assert plugin._action_to_control(np.array([0., -.5]), ego)[0].acceleration < -8.
    np.testing.assert_allclose(plugin._wheelbase(ego), 2.46894)
    np.testing.assert_allclose(old._wheelbase(ego), .6*ego.LENGTH)


def test_speed_dependent_bounds_preserve_supervisor_state_and_target():
    ego = native_car()
    plugin = RearEscapeCBFPlugin(physical_config())
    plugin.filter.supervisor.state = FSMState.ESCAPE_LEFT
    plugin.filter.supervisor.state_age_steps = 7
    plugin.filter._escape_target = np.array([0., 3.2])
    first = plugin._runtime_config(.3, ego)
    np.testing.assert_allclose(first.max_curvature, .05)
    assert first.max_acceleration < 3.
    ego.velocity = np.array([20., 0.])
    second = plugin._runtime_config(.3, ego)
    np.testing.assert_allclose(second.max_curvature, .0125)
    assert plugin.filter.supervisor.state is FSMState.ESCAPE_LEFT
    assert plugin.filter.supervisor.state_age_steps == 7
    np.testing.assert_array_equal(plugin.filter._escape_target, [0., 3.2])


def test_lateral_tracking_cost_has_same_units_at_different_speeds():
    plugin = RearEscapeCBFPlugin(physical_config())
    for speed in [5., 10., 20.]:
        runtime = plugin._runtime_config(.3, native_car(speed))
        one_mps2_as_curvature = 1. / speed**2
        np.testing.assert_allclose(runtime.curvature_weight*one_mps2_as_curvature**2, 4.)


def test_preconditioner_keeps_hard_qp_solution_identical():
    args = dict(reference=np.array([1., .1]), weights=np.array([1., 1e4]),
                constraints=[LinearConstraint(np.array([1., 2.]), .4, 'vehicle', 'vehicle')],
                lower_bounds=np.array([-6., -.3]), upper_bounds=np.array([3., .3]))
    ordinary = solve_prioritized_projection(**args)
    scaled = solve_prioritized_projection(**args, precondition=True)
    assert ordinary.hard_feasible and scaled.hard_feasible
    np.testing.assert_allclose(ordinary.control, scaled.control, atol=1e-8)
    np.testing.assert_allclose(ordinary.objective, scaled.objective, atol=1e-8)


def test_preconditioned_infeasible_fallback_reaches_known_bounded_optimum():
    args = dict(reference=np.array([0., .05]), weights=np.array([1., 1e4]),
        constraints=[LinearConstraint(np.array([1., 0.]), 4., 'rear', 'vehicle'),
                     LinearConstraint(np.array([0., -1.]), 0., 'left', 'road')],
        lower_bounds=np.array([-6., -.3]), upper_bounds=np.array([3., .3]))
    result = solve_prioritized_projection(**args, precondition=True)
    assert not result.hard_feasible and result.mandatory_feasible
    np.testing.assert_allclose(result.control, [3., 0.], atol=1e-7)


@pytest.mark.parametrize('heading', [-.15, 0., .15])
@pytest.mark.parametrize('speed', [.5, 3., 10.])
def test_velocity_corner_barrier_derivative_and_geometric_implication(heading, speed):
    cbf = RearEscapeCBF(physical_config())
    ego = VehicleState2D('ego', np.zeros(2), speed*np.array([np.cos(heading),np.sin(heading)]),
                         np.zeros(2), heading, 4.515, 1.852)
    road = RoadGeometry(0., 5., 5.)
    u = np.array([.7, .02])
    dt = 1e-7
    e = ego.heading_vector
    l = np.array([-e[1], e[0]])
    future = replace(ego, position=ego.position+dt*ego.velocity,
                     velocity=ego.velocity+dt*(u[0]*e+speed**2*u[1]*l),
                     heading=ego.heading+dt*speed*u[1])
    future_road = RoadGeometry(0., 5.-dt*ego.velocity[1], 5.+dt*ego.velocity[1])
    previous = replace(ego, position=ego.position-dt*ego.velocity,
                       velocity=ego.velocity-dt*(u[0]*e+speed**2*u[1]*l),
                       heading=ego.heading-dt*speed*u[1])
    previous_road = RoadGeometry(0., 5.+dt*ego.velocity[1], 5.-dt*ego.velocity[1])
    before = cbf._road_barriers(ego, road)
    after = cbf._road_barriers(future, future_road)
    earlier = cbf._road_barriers(previous, previous_road)
    for a, b, c in zip(before, after, earlier):
        assert a.diagnostic.barrier_model == ROAD_MODEL
        assert a.diagnostic.h <= a.diagnostic.geometric_gap + 1e-10
        finite_difference = (b.diagnostic.h-c.diagnostic.h)/(2.*dt)
        analytic = a.diagnostic.h_dot + a.constraint.coefficients@u
        np.testing.assert_allclose(analytic, finite_difference, atol=5e-5)


def test_no_obstacle_passthrough_cannot_bypass_physical_curvature_bound():
    cbf = RearEscapeCBF(physical_config(max_curvature=.05))
    ego = VehicleState2D('ego', np.zeros(2), np.array([10., 0.]), np.zeros(2), 0., 4.515, 1.852)
    result = cbf.filter(Control2D(0., .2), ego, [], RoadGeometry(0., 5., 5.))
    assert result.qp_solve_count == 1
    assert abs(result.control.curvature) <= .05+1e-8


def test_nonintervened_native_plugin_preserves_policy_action():
    ego = native_car()
    env = SimpleNamespace(agent=ego, engine=SimpleNamespace(
        traffic_manager=SimpleNamespace(vehicles=[])))
    plugin = RearEscapeCBFPlugin(physical_config())
    plugin.reset(env)
    for raw in [np.array([.01, .1]), np.array([0., -.1])]:
        action, telemetry = plugin.filter_action(env, raw)
        assert not telemetry['rta_intervened']
        np.testing.assert_allclose(action, raw, atol=1e-8)


def guard_env(ego, others=()):
    return SimpleNamespace(agent=ego, engine=SimpleNamespace(
        traffic_manager=SimpleNamespace(vehicles=[ego, *others])))


def test_rear_guard_is_exactly_legacy_when_no_rear_threat_exists():
    from cbf_rta.adaptive_plugin import FSMHOCBFPlugin
    ego = native_car()
    env = guard_env(ego)
    legacy_config = CBFConfig(road_margin=.05, passthrough_without_active_obstacle=True)
    guard = RearPhysicalGuardPlugin(physical_config(), legacy_config)
    legacy = FSMHOCBFPlugin(legacy_config)
    guard.reset(env)
    legacy.reset(env)
    for raw in [np.array([0., .2]), np.array([.05, -.1])]:
        a, metadata = guard.filter_action(env, raw)
        b, old = legacy.filter_action(env, raw)
        np.testing.assert_array_equal(a, b)
        assert not metadata['rta_physical_guard_active']
        assert metadata['rta_mode'] == old['rta_mode']


def test_rear_guard_activates_on_approach_and_executes_only_one_filter(monkeypatch):
    ego, rear = native_car(), native_car(15.)
    rear.name, rear.position = 'rear', np.array([-12., 0.])
    env = guard_env(ego, [rear])
    guard = RearPhysicalGuardPlugin(physical_config(), CBFConfig(road_margin=.05))
    guard.reset(env)
    def forbidden(*args, **kwargs):
        raise AssertionError('Unselected legacy controller must not solve a QP')
    monkeypatch.setattr(guard.legacy, 'filter_action', forbidden)
    _, metadata = guard.filter_action(env, np.array([0., 0.]))
    assert metadata['rta_physical_guard_active']
    assert metadata['rta_selected_plugin'] == 'fsm_rear_escape_cbf_qp'


def test_rear_guard_does_not_confuse_an_adjacent_vehicle_with_an_attacker():
    ego, side = native_car(), native_car(13.)
    side.name, side.position = 'side', np.array([-.8, 3.8])
    env = guard_env(ego, [side])
    guard = RearPhysicalGuardPlugin(physical_config(), CBFConfig(road_margin=.05))
    guard.reset(env)
    _, metadata = guard.filter_action(env, np.array([0., 0.]))
    assert not metadata['rta_physical_guard_active']


def test_rear_guard_checks_physical_road_safe_set_before_handoff():
    ego, rear = native_car(), native_car(15.)
    ego.dist_to_left_side = .95
    rear.name, rear.position = 'rear', np.array([-12., 0.])
    env = guard_env(ego, [rear])
    guard = RearPhysicalGuardPlugin(physical_config(), CBFConfig(road_margin=.05))
    guard.reset(env)
    _, metadata = guard.filter_action(env, np.array([0., 0.]))
    assert not metadata['rta_physical_guard_active']
    assert metadata['rta_rear_guard_reason'] == 'rear_approach_outside_physical_road_set'


def test_rear_guard_returns_to_legacy_only_after_stable_clearance():
    ego, rear = native_car(), native_car(15.)
    rear.name, rear.position = 'rear', np.array([-12., 0.])
    env = guard_env(ego, [rear])
    guard = RearPhysicalGuardPlugin(physical_config(), CBFConfig(road_margin=.05))
    guard.reset(env)
    guard.filter_action(env, np.array([0., 0.]))
    assert guard.active
    rear.position = np.array([20., 0.])
    for _ in range(guard.config.fsm_clear_steps):
        guard.filter_action(env, np.array([0., 0.]))
    assert not guard.active


def sensed_trajectory_car(cloud):
    ego = native_car()
    ego.dist_to_left_side, ego.dist_to_right_side = 1., 3.
    ego.navigation = type('TrajectoryNavigation', (), {'current_heading_theta_at_long': 0.})()
    ego.config['side_detector'] = dict(num_lasers=12, distance=50.)
    ego.engine.physics_world = SimpleNamespace(static_world=object())
    ego.side_detector = SimpleNamespace(start_phase_offset=np.pi/2.,
        perceive=lambda *args, **kwargs: SimpleNamespace(cloud_points=np.asarray(cloud)))
    return ego


def test_physical_geometry_does_not_treat_synthetic_route_as_road_edge():
    ego = sensed_trajectory_car(np.ones(12))
    plugin = RearEscapeCBFPlugin(physical_config(rear_use_physical_road_geometry=True))
    road = plugin._road_geometry(ego)
    assert road.left_clearance > 40. and road.right_clearance > 40.
    assert plugin._road_geometry_source == 'continuous_line_sidewalk_sensor'
    old = RearEscapeCBFPlugin(physical_config())._road_geometry(ego)
    assert old.left_clearance == 1. and old.right_clearance == 3.


def test_physical_geometry_keeps_real_continuous_line_limits():
    cloud = np.ones(12)
    cloud[0], cloud[6] = .06, .12
    ego = sensed_trajectory_car(cloud)
    plugin = RearEscapeCBFPlugin(physical_config(rear_use_physical_road_geometry=True))
    road = plugin._road_geometry(ego)
    np.testing.assert_allclose([road.left_clearance, road.right_clearance], [3., 6.])


def test_physical_geometry_sensor_failure_does_not_remove_road_limits():
    ego = sensed_trajectory_car(np.ones(12))
    ego.config['side_detector']['num_lasers'] = 0
    plugin = RearEscapeCBFPlugin(physical_config(rear_use_physical_road_geometry=True))
    road = plugin._road_geometry(ego)
    assert road.left_clearance == 1. and road.right_clearance == 3.
    assert 'unavailable_fallback' in plugin._road_geometry_source


def test_oblique_sensor_line_fit_recovers_normal_and_transverse_clearance():
    normal = np.array([.3, -np.sqrt(.91)])
    tangent = np.array([-normal[1], normal[0]])
    anchor = -3.*normal
    points = [anchor-2.*tangent, anchor+tangent, anchor+3.*tangent]
    clearance, fitted = RearEscapeCBFPlugin._fit_side_plane(points, np.array([0., -1.]))
    np.testing.assert_allclose(fitted, normal, atol=1e-10)
    np.testing.assert_allclose(clearance, 3./np.sqrt(.91))
    assert RearEscapeCBFPlugin._fit_side_plane([anchor], np.array([0., -1.])) is None


def test_sensed_corner_gap_overrides_parallel_center_approximation():
    cbf = RearEscapeCBF(physical_config())
    ego = VehicleState2D('e', np.zeros(2), np.array([10., 0.]), np.zeros(2), 0., 4.5, 1.85)
    name = 'road:left:front_left'
    road = RoadGeometry(0., 5., 5., corner_boundaries={name:(.3, np.array([0., -1.]))})
    barrier = next(b for b in cbf._road_barriers(ego, road) if b.diagnostic.name == name)
    np.testing.assert_allclose(barrier.diagnostic.geometric_gap, .25)


def test_guard_low_speed_close_rear_decision_is_not_suppressed_by_receding_speed():
    ego, rear = native_car(.5), native_car(-.7)
    rear.name, rear.position = 'rear', np.array([-10., 0.])
    env = guard_env(ego, [rear])
    guard = RearPhysicalGuardPlugin(physical_config(rear_guard_allow_slow_approach=True),
                                    CBFConfig(road_margin=.05))
    guard.reset(env)
    _, metadata = guard.filter_action(env, np.array([0., 1.]))
    assert metadata['rta_physical_guard_active']


def test_procedural_road_frame_does_not_rotate_with_ego_heading():
    ego = native_car()
    ego.heading_theta = .3
    lane = SimpleNamespace(local_coordinates=lambda position: (0., 0.),
                           heading_theta_at=lambda longitudinal: 0.)
    ego.navigation = SimpleNamespace(current_ref_lanes=[lane])
    plugin = RearEscapeCBFPlugin(physical_config(rear_use_physical_road_geometry=True))
    assert plugin._road_geometry(ego).heading == 0.
    # Existing profiles retain their previous convention.
    assert RearEscapeCBFPlugin(physical_config())._road_geometry(ego).heading == .3


def test_guard_handoff_checks_destination_road_geometry(monkeypatch):
    ego, rear = native_car(), native_car(15.)
    rear.name, rear.position = 'rear', np.array([-12., 0.])
    env = guard_env(ego, [rear])
    guard = RearPhysicalGuardPlugin(physical_config(rear_guard_consistent_handoff=True),
                                    CBFConfig(road_margin=.05))
    guard.reset(env)
    guard.filter_action(env, np.array([0., 0.]))
    assert guard.active
    rear.position = np.array([25., 0.])
    # Physical road is clear, but the receiving legacy controller would use
    # an artificial road edge cutting through the ego body. Do not switch.
    monkeypatch.setattr(guard.physical, '_road_geometry', lambda obj: RoadGeometry(0., 5., 5.))
    monkeypatch.setattr(guard.legacy, '_road_geometry', lambda obj: RoadGeometry(0., -.5, 5.))
    for _ in range(guard.config.fsm_clear_steps+1):
        _, telemetry = guard.filter_action(env, np.array([0., 0.]))
    assert guard.active
    assert not telemetry['rta_handoff_safe']
    assert telemetry['rta_handoff_min_road_h'] < 0.
    monkeypatch.setattr(guard.legacy, '_road_geometry', lambda obj: RoadGeometry(0., 5., 5.))
    for _ in range(guard.config.fsm_clear_steps):
        guard.filter_action(env, np.array([0., 0.]))
    assert not guard.active


def test_long_closing_ttc_does_not_request_full_physical_escape():
    ego, rear = native_car(), native_car(10.3)
    rear.name, rear.position = 'rear', np.array([-10., 0.])
    env = guard_env(ego, [rear])
    gated = RearPhysicalGuardPlugin(physical_config(rear_guard_risk_gated=True, rear_trigger_ttc=3.),
                                    CBFConfig(road_margin=.05))
    original = RearPhysicalGuardPlugin(physical_config(rear_trigger_ttc=3.), CBFConfig(road_margin=.05))
    gated.reset(env)
    original.reset(env)
    _, metadata = gated.filter_action(env, np.array([0., 0.]))
    original.filter_action(env, np.array([0., 0.]))
    assert original.active and not gated.active
    assert metadata['rta_rear_guard_reason'] == 'legacy_nonurgent_rear_risk'
    assert metadata['rta_enabled']
    assert metadata['rta_selected_plugin'] == 'fsm_hocbf_qp'
    rear.velocity = np.array([15., 0.])
    _, metadata = gated.filter_action(env, np.array([0., 0.]))
    assert gated.active


def test_nominal_drive_limit_is_native_not_a_fixed_three_mps2_cap():
    ego = native_car()
    ego.config['max_engine_force'] = 900.
    plugin = RearEscapeCBFPlugin(physical_config(max_acceleration=4.))
    runtime = plugin._runtime_config(.3, ego)
    np.testing.assert_allclose(runtime.max_acceleration, 4.*900./ego.MASS)
    nominal, limit = plugin._action_to_control(np.array([0., 1.]), ego)
    action = plugin._control_to_action(Control2D(runtime.max_acceleration, 0.), ego, limit)
    np.testing.assert_allclose(action[1], 1.)


def test_frenet_geometry_reads_local_route_curvature_and_lateral_offset():
    ego = sensed_trajectory_car(np.ones(12))
    ego.position = np.array([4., 7.])
    # The heading derivative crosses the angle wrap; curvature stays +0.01/m.
    trajectory = SimpleNamespace(length=100., local_coordinates=lambda position: (10., 2.),
        position=lambda longitudinal, lateral: np.array([4., 5.]),
        heading_theta_at=lambda longitudinal: np.arctan2(
            np.sin(np.pi-.1+.01*longitudinal), np.cos(np.pi-.1+.01*longitudinal)))
    ego.navigation.reference_trajectory = trajectory
    plugin = RearEscapeCBFPlugin(physical_config(rear_use_physical_road_geometry=True,
                                                rear_frenet_escape=True))
    road = plugin._road_geometry(ego)
    np.testing.assert_allclose(road.reference_lateral_error, -2.)
    np.testing.assert_allclose(road.reference_curvature, .01, atol=1e-10)


def test_valid_physical_road_can_correct_artificial_legacy_fail_stop(monkeypatch):
    ego = native_car(1.)
    env = guard_env(ego)
    guard = RearPhysicalGuardPlugin(physical_config(rear_guard_correct_road_domain=True,
                                                    rear_guard_consistent_handoff=True,
                                                    rear_use_physical_road_geometry=True),
                                    CBFConfig(road_margin=.05))
    guard.reset(env)
    monkeypatch.setattr(guard.physical, '_road_geometry', lambda obj: RoadGeometry(0., 5., 5.))
    monkeypatch.setattr(guard.legacy, '_road_geometry', lambda obj: RoadGeometry(0., -.5, 5.))
    _, telemetry = guard.filter_action(env, np.array([0., .5]))
    assert guard.active
    assert telemetry['rta_legacy_road_domain_conflict']
    assert telemetry['rta_rear_guard_reason'] == 'physical_road_valid_legacy_domain_conflict'
    assert telemetry['rta_safe_acceleration'] > 0.
    # This is not an unconditional road-constraint relaxation.
    guard.reset(env)
    monkeypatch.setattr(guard.physical, '_road_geometry', lambda obj: RoadGeometry(0., -.5, 5.))
    guard.filter_action(env, np.array([0., .5]))
    assert not guard.active
    # Domain correction is limited to the legacy low-speed fail-stop branch.
    # It must not broaden physical takeover during normal high-speed driving.
    guard.reset(env)
    ego.velocity = np.array([10., 0.])
    monkeypatch.setattr(guard.physical, '_road_geometry', lambda obj: RoadGeometry(0., 5., 5.))
    guard.filter_action(env, np.array([0., .5]))
    assert not guard.active
