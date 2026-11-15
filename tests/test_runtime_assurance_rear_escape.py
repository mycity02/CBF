"""Geometry, dynamics, isolation and adapter regressions for rear protection."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from bmt.runtime_assurance.cbf2d import TwoDimensionalCBF
from bmt.runtime_assurance.adaptive_plugin import RearEscapeCBFPlugin
from bmt.runtime_assurance.fsm import FSMHOCBF, FSMState
from bmt.runtime_assurance.qp import solve_weighted_projection
from bmt.runtime_assurance.rear_escape import REAR_MODEL, RearEscapeCBF, footprint_support
from bmt.runtime_assurance.types import CBFConfig, Control2D, RoadGeometry, VehicleState2D


def car(name, x, y, speed, heading=0., yaw_rate=0., acceleration=(0., 0.)):
    return VehicleState2D(name=name, position=np.array([x, y]),
                          velocity=speed*np.array([np.cos(heading), np.sin(heading)]),
                          acceleration=np.array(acceleration), heading=heading,
                          length=4.6, width=1.85, yaw_rate=yaw_rate)


def config(**kwargs):
    values = dict(other_accel_uncertainty=0., rear_yaw_rate_uncertainty=0.,
                  rear_velocity_barrier_enabled=True)
    values.update(kwargs)
    return CBFConfig(**values)


def test_support_superellipse_contains_all_body_corners():
    ego = car('ego', 0., 0., 10., .31)
    e = ego.heading_vector
    l = np.array([-e[1], e[0]])
    for angle in np.linspace(-np.pi, np.pi, 91):
        n = np.array([np.cos(angle), np.sin(angle)])
        radius = footprint_support(ego, n)[0]
        assert all(n @ (s*.5*ego.length*e+t*.5*ego.width*l) <= radius+1e-10
                   for s in [-1., 1.] for t in [-1., 1.])


@pytest.mark.parametrize('mode', [-1., 0., 1.])
@pytest.mark.parametrize('other_speed', [6., 15.])
def test_rear_cbf_exact_derivative_includes_normal_and_body_rotation(mode, other_speed):
    cbf = RearEscapeCBF(config())
    ego = car('ego', 1., .3, 10., .17)
    other = car('rear', -11., -.4, other_speed, -.08, .14, (.2, -.3))
    axis = np.array([0., 1.])
    barrier = cbf._rear_support_barrier(ego, other, axis, mode)
    u = np.array([.7, .03])
    dt = 1e-7
    e = ego.heading_vector
    l = np.array([-e[1], e[0]])
    next_ego = replace(ego, position=ego.position+dt*ego.velocity,
                       velocity=ego.velocity+dt*(u[0]*e+ego.speed**2*u[1]*l),
                       heading=ego.heading+dt*ego.speed*u[1])
    next_other = replace(other, position=other.position+dt*other.velocity,
                         velocity=other.velocity+dt*other.acceleration,
                         heading=other.heading+dt*other.yaw_rate)
    value = cbf._rear_support_barrier(next_ego, next_other, axis, mode).diagnostic.h
    finite_difference = (value-barrier.diagnostic.h)/dt
    analytic = barrier.diagnostic.h_dot + barrier.constraint.coefficients @ u
    np.testing.assert_allclose(analytic, finite_difference, atol=1e-5)


def test_directional_rear_barrier_restores_steering_authority_in_centered_case():
    cfg = config(road_margin=.05)
    legacy = TwoDimensionalCBF(cfg)
    cbf = RearEscapeCBF(cfg)
    ego, other = car('ego', 0., 0., 10.), car('rear', -12., 0., 15.)
    road = RoadGeometry(0., 5., 5.)
    old = legacy._vehicle_barrier(ego, other, road)
    assert old.constraint.coefficients[1] == 0.
    assert old.constraint.lower/old.constraint.coefficients[0] > cfg.max_acceleration
    new = cbf._rear_support_barrier(ego, other, road.left_normal, 1.)
    assert new.diagnostic.h > 0.
    assert new.constraint.coefficients[1] > 0.
    solution = solve_weighted_projection(np.array([0., .04]), np.array([1., 40.]),
                                         [new.constraint], np.array([-6., -.3]), np.array([3., .3]))
    assert solution.hard_feasible
    assert solution.control[1] > 0.


def test_left_right_constraints_are_mirror_symmetric():
    cbf = RearEscapeCBF(config())
    left = cbf._rear_support_barrier(car('e', 0., .2, 10., .1), car('r', -12., -.4, 15., -.04),
                                    np.array([0., 1.]), 1.)
    right = cbf._rear_support_barrier(car('e', 0., -.2, 10., -.1), car('r', -12., .4, 15., .04),
                                     np.array([0., 1.]), -1.)
    np.testing.assert_allclose(left.diagnostic.h, right.diagnostic.h)
    np.testing.assert_allclose(left.constraint.lower, right.constraint.lower)
    np.testing.assert_allclose(left.constraint.coefficients, right.constraint.coefficients*np.array([1., -1.]))


def test_centimeters_of_space_are_not_a_valid_escape_corridor():
    cbf = RearEscapeCBF(config(road_margin=.05))
    ego, other = car('e', 0., 0., 10.), car('r', -12., 0., 15.)
    assert cbf._direction_margin(1., ego, [other], RoadGeometry(0., 1., 5.)) < 0.
    assert cbf._direction_margin(-1., ego, [other], RoadGeometry(0., 1., 5.)) > 0.


def test_filter_selects_open_side_and_replaces_instead_of_duplicating_rear_constraint():
    cbf = RearEscapeCBF(config(road_margin=.05))
    result = cbf.filter(Control2D(0., 0.), car('e', 0., 0., 10.), [car('r', -12., 0., 15.)],
                        RoadGeometry(0., 1., 5.))
    assert result.mode == FSMState.ESCAPE_RIGHT.value
    rear = [b for b in result.diagnostics if b.name == 'vehicle:r']
    assert len(rear) == 1
    assert rear[0].barrier_model == REAR_MODEL
    assert result.hard_feasible
    assert result.control.curvature < 0.
    assert result.qp_solve_count == 1
    assert result.candidate_scores == {}


def test_escape_target_is_latched_and_reset_clears_it():
    cbf = RearEscapeCBF(config(road_margin=.05))
    ego, other = car('e', 0., 0., 10.), car('r', -12., 0., 15.)
    road = RoadGeometry(0., 5., 5.)
    cbf.filter(Control2D(0., 0.), ego, [other], road)
    target = cbf._escape_target.copy()
    cbf.filter(Control2D(0., 0.), replace(ego, position=np.array([1., .1])),
               [replace(other, position=np.array([-10.5, 0.]))], road)
    np.testing.assert_allclose(cbf._escape_target, target)
    cbf.reset()
    assert cbf._escape_target is None
    assert cbf._rear_axes == {}


def test_no_rear_threat_preserves_legacy_output():
    cfg = config(road_margin=.05, passthrough_without_active_obstacle=True)
    ego = car('e', 0., 0., 10.)
    road = RoadGeometry(0., 5., 5.)
    nominal = Control2D(1., .01)
    old = FSMHOCBF(cfg).filter(nominal, ego, [], road)
    new = RearEscapeCBF(cfg).filter(nominal, ego, [], road)
    assert old.control == new.control
    assert old.mode == new.mode
    assert old.qp_solve_count == new.qp_solve_count


def test_positive_velocity_barrier_implies_positive_geometric_separation():
    cbf = RearEscapeCBF(config())
    for y in [-2., 0., 2.]:
        for speed in [8., 15.]:
            barrier = cbf._rear_support_barrier(car('e', 0., y, 10.), car('r', -12., 0., speed),
                                               np.array([0., 1.]), 1.)
            assert barrier.diagnostic.h <= barrier.diagnostic.geometric_gap+1e-10


@pytest.mark.parametrize('values', [dict(rear_normal_bias=0.), dict(rear_relative_accel_scale=0.),
                                    dict(rear_yaw_rate_uncertainty=-1.), dict(estimated_yaw_rate_limit=0.)])
def test_invalid_rear_parameters_are_rejected(values):
    with pytest.raises(ValueError):
        CBFConfig(**values)


def test_adjacent_car_with_center_behind_does_not_define_rear_escape_target():
    cbf = RearEscapeCBF(config())
    ego, side = car('ego', 0., 0., 10.), car('side', -.8, 3.8, 13.)
    assert cbf._rear_candidates(ego, [side], RoadGeometry(0., 5., 5.)) == []


def test_partial_lateral_room_is_not_reported_as_complete_escape():
    cbf = RearEscapeCBF(config(road_margin=.05))
    ego, rear = car('ego', 0., 0., 10.), car('rear', -12., 0., 15.)
    result = cbf.filter(Control2D(0., 0.), ego, [rear], RoadGeometry(0., 1., 3.))
    assert result.mode == 'escape_right'
    assert not cbf._escape_complete_corridor
    assert abs(cbf._escape_target[1]) >= cbf.config.rear_min_lateral_shift
    assert abs(cbf._escape_target[1]) < 3.-ego.width/2.-cbf.config.road_margin


def test_directional_normal_remains_well_defined_when_longitudinally_alongside():
    cbf = RearEscapeCBF(config())
    barrier = cbf._rear_support_barrier(car('ego', 0., -4., 10.), car('rear', 0., 0., 10.),
                                       np.array([0., 1.]), 1.)
    assert np.all(np.isfinite(barrier.constraint.coefficients))
    assert barrier.diagnostic.geometric_gap > 0.


def test_plugin_tracks_wrapped_yaw_rate_and_resets_history():
    def fake(name, x, speed):
        return SimpleNamespace(name=name, position=np.array([x, 0.]), velocity=np.array([speed, 0.]),
                               heading_theta=0., LENGTH=4.6, WIDTH=1.85, max_steering=40.,
                               dist_to_left_side=5., dist_to_right_side=5.,
                               metadrive_type='vehicle')
    ego, rear = fake('ego', 0., 10.), fake('rear', -12., 15.)
    env = SimpleNamespace(agent=ego, engine=SimpleNamespace(
        traffic_manager=SimpleNamespace(vehicles=[ego, rear])))
    plugin = RearEscapeCBFPlugin(config(road_margin=.05))
    plugin.reset(env)
    action, telemetry = plugin.filter_action(env, np.array([0., 0.]))
    assert np.all(np.abs(action) <= 1.)
    assert telemetry['rta_rear_barrier_count'] == 1
    assert telemetry['rta_plugin'] == 'fsm_rear_escape_cbf_qp'
    rear.heading_theta = np.pi-.1
    plugin._remember_velocities([ego, rear])
    rear.heading_theta = -np.pi+.1
    np.testing.assert_allclose(plugin._vehicle_state(rear).yaw_rate, 2.)
    plugin.reset(env)
    assert plugin._vehicle_state(rear).yaw_rate == 0.


def test_barrier_ablation_keeps_position_target_but_restores_original_hocbf():
    cbf = RearEscapeCBF(config(road_margin=.05, rear_velocity_barrier_enabled=False))
    result = cbf.filter(Control2D(0., 0.), car('e', 0., 0., 10.), [car('r', -12., 0., 15.)],
                        RoadGeometry(0., 1., 5.))
    assert result.mode == 'escape_right'
    assert cbf._escape_target is not None
    assert all(b.barrier_model != REAR_MODEL for b in result.diagnostics)


def test_experimental_barrier_stays_opt_in_due_to_low_speed_steering_limit():
    cbf = RearEscapeCBF(config(rear_normal_bias=8., rear_relative_accel_scale=6.))
    barrier = cbf._rear_support_barrier(car('ego', 0., 0., .5), car('rear', -12., 0., 5.5),
                                       np.array([0., 1.]), 1.)
    # At low speed the footprint-rotation term can oppose the requested side.
    # This capability limit is why the experimental constraint is NOT default.
    assert barrier.constraint.coefficients[1] < 0.
    assert not CBFConfig().rear_velocity_barrier_enabled


def test_instantaneous_ttc_can_trigger_before_legacy_h_threshold():
    cbf = RearEscapeCBF(config(rear_trigger_ttc=3.))
    ego, rear = car('e', 0., 0., 10.), car('r', -25., 0., 20.)
    road = RoadGeometry(0., 5., 5.)
    assert cbf._rear_entering(ego, rear, road, SimpleNamespace(h=10., psi1=10.))
    assert not cbf._rear_entering(ego, replace(rear, velocity=np.array([8., 0.])), road,
                                  SimpleNamespace(h=10., psi1=10.))


def test_ongoing_corridor_checks_remaining_distance_not_new_minimum_shift():
    cfg = config(rear_latched_corridor=True, rear_velocity_barrier_enabled=False, road_margin=.05)
    cbf = RearEscapeCBF(cfg)
    cbf.supervisor.state = FSMState.ESCAPE_LEFT
    cbf._escape_target, cbf._escape_axis = np.array([0., 2.2]), np.array([0., 1.])
    ego, rear = car('e', 0., 1.7, 10.), car('r', -8., 0., 15.)
    road = RoadGeometry(0., 1.7, 5.)
    assert cbf._bounded_target(1., ego, [rear], road)[2] < 0.
    assert cbf._direction_margin(1., ego, [rear], road) > 0.
    # A newly occupied path still cancels the maneuver; latching is not blind commitment.
    side = car('side', 0., 3.5, 10.)
    assert cbf._direction_margin(1., ego, [rear, side], road) < 0.


def test_rear_gain_override_matches_direct_hocbf_construction():
    cfg = config(rear_hocbf_gain=2.5, rear_velocity_barrier_enabled=False)
    cbf = RearEscapeCBF(cfg)
    ego, rear, front = car('e', 0., 0., 10.), car('r', -12., .3, 15.), car('f', 12., .3, 8.)
    road = RoadGeometry(0., 5., 5.)
    expected = TwoDimensionalCBF(replace(cfg, alpha1=2.5, alpha2=2.5))._vehicle_barrier(ego, rear, road)
    actual = cbf._vehicle_barrier(ego, rear, road)
    np.testing.assert_allclose(actual.constraint.lower, expected.constraint.lower)
    np.testing.assert_allclose(actual.diagnostic.psi1, expected.diagnostic.psi1)
    # No hidden change to front-vehicle gains.
    np.testing.assert_allclose(cbf._vehicle_barrier(ego, front, road).constraint.lower,
                              TwoDimensionalCBF(cfg)._vehicle_barrier(ego, front, road).constraint.lower)


def test_safe_rejoin_reference_only_runs_after_risks_clear():
    from bmt.runtime_assurance.fsm import ThreatAssessment
    cbf = RearEscapeCBF(config(rear_return_to_reference=True, rear_velocity_barrier_enabled=False))
    cbf._rear_recovery_pending = True
    ego = car('e', 0., 2., 10.)
    road = RoadGeometry(0., 3., 7., reference_lateral_error=-2.)
    nominal = Control2D(1., 0.)
    clear = ThreatAssessment(False, False, True, True, 1., 1.)
    reference = cbf._reference_for_state(FSMState.RECOVERY, nominal, ego, road, clear)
    assert reference.curvature < 0.
    np.testing.assert_allclose(reference.acceleration, 1.)
    assert not cbf._allow_inactive_passthrough()
    cbf.reset()
    assert cbf._allow_inactive_passthrough()


def test_missing_route_reference_does_not_latch_unfinishable_recovery():
    cbf = RearEscapeCBF(config(rear_return_to_reference=True, rear_velocity_barrier_enabled=False))
    cbf.filter(Control2D(0., 0.), car('e', 0., 0., 10.), [car('r', -12., 0., 15.)],
               RoadGeometry(0., 5., 5.))
    assert not cbf._rear_recovery_pending


def test_crossing_vehicle_behind_is_not_a_following_rear_attacker():
    cbf = RearEscapeCBF(config(rear_require_aligned_heading=True))
    ego, cross = car('e', 0., 0., 10.), car('cross', -8., 0., 12., np.pi/2.)
    assert cbf._rear_candidates(ego, [cross], RoadGeometry(0., 5., 5.)) == []
    # Classification does not delete the actor from the generic vehicle CBF path.
    assert cbf._vehicle_barrier(ego, cross, RoadGeometry(0., 5., 5.)).diagnostic.kind == 'vehicle'


def test_longitudinal_reserve_avoids_unnecessary_rear_escape_without_removing_cbf():
    cfg = config(rear_prefer_longitudinal_viability=True, rear_velocity_barrier_enabled=False,
                 rear_trigger_ttc=3., road_margin=.05)
    cbf = RearEscapeCBF(cfg)
    ego, rear = car('e', 0., 0., 10.), car('r', -11., 0., 10.5)
    result = cbf.filter(Control2D(1., 0.), ego, [rear], RoadGeometry(0., 5., 5.))
    assert cbf._rear_longitudinal_hold
    assert result.mode == 'nominal'
    assert any(b.name == 'vehicle:r' for b in result.diagnostics)
    assert result.qp_solve_count == 1
    assert result.hard_feasible


def test_urgent_rear_or_front_conflict_does_not_enable_longitudinal_hold():
    cfg = config(rear_prefer_longitudinal_viability=True, rear_velocity_barrier_enabled=False,
                 rear_trigger_ttc=3., road_margin=.05)
    cbf = RearEscapeCBF(cfg)
    ego, fast = car('e', 0., 0., 10.), car('r', -11., 0., 20.)
    result = cbf.filter(Control2D(1., 0.), ego, [fast], RoadGeometry(0., 5., 5.))
    assert not cbf._rear_longitudinal_hold
    assert result.mode in ('escape_left', 'escape_right')
    slow, front = car('r', -11., 0., 10.5), car('f', 7., 0., 0.)
    assert not cbf._longitudinal_hold_is_feasible(ego, [slow], [slow, front], RoadGeometry(0., 5., 5.))


def test_longitudinal_hold_requires_acceleration_reserve_and_respects_commitment():
    cfg = config(rear_prefer_longitudinal_viability=True, rear_velocity_barrier_enabled=False,
                 rear_trigger_ttc=3., road_margin=.05)
    cbf = RearEscapeCBF(cfg)
    ego, slow = car('e', 0., 0., 10.), car('r', -11., 0., 10.5)
    road = RoadGeometry(0., 5., 5.)
    assert cbf._longitudinal_hold_is_feasible(ego, [slow], [slow], road)
    accelerating = replace(slow, acceleration=np.array([5., 0.]))
    assert not cbf._longitudinal_hold_is_feasible(ego, [accelerating], [accelerating], road)
    cbf.supervisor.state = FSMState.ESCAPE_LEFT
    assert not cbf._longitudinal_hold_is_feasible(ego, [slow], [slow], road)


def test_frenet_escape_tracks_road_offset_not_old_world_position():
    cfg = config(rear_frenet_escape=True, rear_velocity_barrier_enabled=False,
                 rear_latched_corridor=True, rear_use_native_dynamics=True)
    cbf = RearEscapeCBF(cfg)
    cbf.supervisor.state = FSMState.ESCAPE_LEFT
    cbf._escape_target = np.array([0., 3.2])
    cbf._escape_axis = np.array([0., 1.])
    cbf._escape_lateral_target = 3.2
    # Far along a curved road, the same lateral target has already been reached.
    # Current velocity is tangent to this road, not to the initial world axis.
    road = RoadGeometry(.5, 5., 5., reference_lateral_error=-3.2, reference_curvature=.02)
    ego = car('e', 80., 30., 10., .5)
    reference = cbf._reference_for_state(FSMState.ESCAPE_LEFT, Control2D(1., 0.), ego, road)
    np.testing.assert_allclose(reference.curvature, .02, atol=1e-10)
    # A global translation must not change the control at the same Frenet state.
    shifted = replace(ego, position=ego.position+np.array([200., -150.]))
    np.testing.assert_allclose(cbf._reference_for_state(FSMState.ESCAPE_LEFT, Control2D(1., 0.),
                                                       shifted, road).as_array(), reference.as_array())
    cbf.reset()
    assert cbf._escape_lateral_target is None


def test_cleared_rear_maneuver_transitions_to_front_brake_without_dropping_constraints():
    from bmt.runtime_assurance.fsm import SafetySupervisorFSM, ThreatAssessment
    supervisor = SafetySupervisorFSM(config(rear_release_completed_escape=True))
    supervisor.state, supervisor.state_age_steps = FSMState.ESCAPE_LEFT, 10
    assessment = ThreatAssessment(front_threat=True, rear_threat=False, front_clear=False,
                                  rear_clear=True, left_margin=1., right_margin=1.)
    decision = supervisor.update(assessment, 0.)
    assert decision.state is FSMState.BRAKE
    assert decision.reason == 'rear_clear_front_brake'
    supervisor.state = FSMState.ESCAPE_LEFT
    decision = supervisor.update(replace(assessment, rear_threat=True, rear_clear=False), 0.)
    assert decision.state is FSMState.ESCAPE_LEFT
