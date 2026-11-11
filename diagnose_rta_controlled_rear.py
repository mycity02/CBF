"""Reproducible MetaDrive rear-escape positive controls and actuator audit.

This is a diagnostic fixture, not a training environment or an RTA planner.
Traffic follows the same prescribed motion in every run. A separate feedback
controller provides a feasibility witness; its actions are never searched or
rolled out by the online safety filter. Output files are never overwritten.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from bmt.runtime_assurance.metadrive_plugin import MetaDriveCBFPlugin
from bmt.runtime_assurance.types import CBFConfig


@dataclass(frozen=True)
class Case:
    name: str
    ego_speed: float = 10.0
    rear_speed: float = 15.0
    rear_distance: float = 12.0
    front_distance: float = 14.0
    duration: float = 6.0
    lateral_target: float = 3.2


def wrap_angle(value):
    return float(np.arctan2(np.sin(value), np.cos(value)))


def reference_action(ego, origin, tangent, normal, target_lateral, target_speed):
    """One fixed PD witness, not a trajectory optimiser or candidate search."""
    displacement = np.asarray(ego.position)[:2] - origin
    velocity = np.asarray(ego.velocity)[:2]
    lateral_error = target_lateral - float(displacement @ normal)
    speed = float(np.linalg.norm(velocity))
    ay = float(np.clip(4.0 * lateral_error - 3.5 * float(velocity @ normal), -5., 5.))
    heading = float(ego.heading_theta)
    left = np.array([-np.sin(heading), np.cos(heading)])
    curvature = ay / (max(speed ** 2, 1.0) * max(float(left @ normal), .25))
    curvature = float(np.clip(curvature, -.20, .20))
    wheelbase = float(ego.FRONT_WHEELBASE + ego.REAR_WHEELBASE)
    steering = np.arctan(curvature * wheelbase) / np.deg2rad(ego.max_steering)
    acceleration = float(np.clip(1.5 * (target_speed - speed), -3., 3.))
    throttle = acceleration / (3.0 if acceleration >= 0. else 6.0)
    return np.clip(np.array([steering, throttle]), -1., 1.)


def build_env(case):
    # Lazy imports keep geometry/report helpers testable without Panda3D.
    from metadrive import MetaDriveEnv
    from metadrive.component.vehicle.vehicle_type import TrafficDefaultVehicle
    from metadrive.manager.base_manager import BaseManager

    class ControlledTraffic(BaseManager):
        PRIORITY = 20  # after the native agent manager has spawned the ego

        def after_reset(self):
            self.clock = 0
            self.contact_roles = set()
            self.contact_trace = []
            ego = self.engine.agent_manager.active_agents['default_agent']
            self.origin = np.asarray(ego.position)[:2].copy()
            self.heading = float(ego.heading_theta)
            self.tangent = np.array([np.cos(self.heading), np.sin(self.heading)])
            self.normal = np.array([-self.tangent[1], self.tangent[0]])
            self.motion = {}
            for role, distance, speed in [('rear', -case.rear_distance, case.rear_speed),
                                          ('front', case.front_distance, case.ego_speed)]:
                if role == 'front' and case.front_distance <= 0.:
                    continue
                v_config = dict(self.engine.global_config['vehicle_config'])
                v_config.update(navigation_module=None, spawn_position_heading=(
                    self.origin + distance * self.tangent, self.heading),
                    spawn_velocity=[0., 0.], no_wheel_friction=True,
                    show_navi_mark=False, show_navigation_arrow=False)
                actor = self.spawn_object(TrafficDefaultVehicle, vehicle_config=v_config,
                                          name='diagnostic_'+role, random_seed=0)
                self.motion[role] = (actor, self.origin+distance*self.tangent, speed)

        @property
        def vehicles(self):
            return list(self.spawned_objects.values())

        def before_step(self):
            for actor, _, _ in self.motion.values():
                actor.before_step([0., 0.])
            return {}

        def capture_contact(self):
            ego = self.engine.agent_manager.active_agents['default_agent']
            world = self.engine.physics_world.dynamic_world
            for role, (actor, _, _) in self.motion.items():
                result = world.contactTestPair(ego.chassis.node(), actor.chassis.node())
                if any(c.getManifoldPoint().getDistance() <= 0. for c in result.getContacts()):
                    if role not in self.contact_roles:
                        self.contact_trace.append({'physics_step': self.clock, 'role': role})
                    self.contact_roles.add(role)

        def step(self):
            # Capture the preceding physics substep BEFORE resetting actor pose.
            self.capture_contact()
            dt = float(self.engine.global_config['physics_world_step_size'])
            for actor, start, speed in self.motion.values():
                actor.set_position(start+self.tangent*speed*self.clock*dt)
                actor.set_heading_theta(self.heading)
                actor.set_velocity(self.tangent*speed)
            self.clock += 1

        def after_step(self):
            self.capture_contact()
            for actor, _, _ in self.motion.values():
                actor.after_step()
            return {}

        def rewind(self):
            self.clock = 0
            self.contact_roles.clear()
            self.contact_trace.clear()
            for actor, start, speed in self.motion.values():
                actor.set_position(start)
                actor.set_heading_theta(self.heading)
                actor.set_velocity(self.tangent*speed)

    class ControlledEnv(MetaDriveEnv):
        def setup_engine(self):
            super().setup_engine()
            self.engine.update_manager('traffic_manager', ControlledTraffic())

    return ControlledEnv(dict(
        use_render=False, show_interface=False, manual_control=False,
        map_config=dict(type='block_sequence', config='S', lane_num=3, lane_width=3.5, exit_length=200),
        traffic_density=0., random_spawn_lane_index=False, num_scenarios=1,
        force_destroy=True,  # prevent suspension/friction state leaking between diagnostic arms
        agent_configs={'default_agent': dict(spawn_lane_index=('>', '>>', 1), spawn_longitude=60.)},
        horizon=int(case.duration/.1)+10, crash_vehicle_done=True,
        log_level=logging.CRITICAL, vehicle_config=dict(show_navi_mark=False,
                                                       show_navigation_arrow=False),
    ))


def initialise(env, case):
    env.reset(seed=0)
    # Settle suspension on the actual road before defining t=0 for all arms.
    for _ in range(3):
        env.step(np.zeros(2))
    traffic = env.engine.traffic_manager
    ego = env.agent
    ego.set_position(traffic.origin)
    ego.set_heading_theta(traffic.heading)
    ego.set_velocity(traffic.tangent*case.ego_speed)
    ego.after_step()
    traffic.rewind()
    return traffic


def vehicle_snapshot(vehicle):
    return dict(position=np.asarray(vehicle.position)[:2].tolist(),
                velocity=np.asarray(vehicle.velocity)[:2].tolist(),
                heading=float(vehicle.heading_theta), length=float(vehicle.LENGTH),
                width=float(vehicle.WIDTH), mass=float(vehicle.MASS),
                actual_wheelbase=float(vehicle.FRONT_WHEELBASE+vehicle.REAR_WHEELBASE),
                max_engine_force=float(vehicle.config['max_engine_force']),
                max_brake_force=float(vehicle.config['max_brake_force']),
                no_wheel_friction=bool(vehicle.config['no_wheel_friction']),
                actual_wheel_friction=float(vehicle.system.getWheel(0).getFrictionSlip()))


def body_road_gap(ego, normal):
    heading = np.array([np.cos(ego.heading_theta), np.sin(ego.heading_theta)])
    lateral = np.array([-heading[1], heading[0]])
    half = .5*(ego.LENGTH*abs(heading@normal)+ego.WIDTH*abs(lateral@normal))
    return min(float(ego.dist_to_left_side), float(ego.dist_to_right_side))-half


def run_arm(env, case, arm, plugin=None):
    traffic = initialise(env, case)
    if plugin is not None:
        plugin.reset(env)
    initial = {'ego': vehicle_snapshot(env.agent),
               **{role: vehicle_snapshot(item[0]) for role, item in traffic.motion.items()}}
    trace = []
    collided = out = False
    min_road_gap = float('inf')
    dt = .1
    for index in range(int(round(case.duration/dt))):
        ego = env.agent
        pre = vehicle_snapshot(ego)
        nominal = reference_action(ego, traffic.origin, traffic.tangent, traffic.normal, 0., case.ego_speed)
        witness = reference_action(ego, traffic.origin, traffic.tangent, traffic.normal,
                                   case.lateral_target, max(case.ego_speed, 8.))
        rta = None
        filtered = nominal
        if plugin is not None:
            filtered, rta = plugin.filter_action(env, nominal)
        executed = witness if arm.startswith('witness') else filtered
        excluded_by = []
        if plugin is not None and arm.startswith('witness'):
            physical, _ = plugin._action_to_control(executed, ego)
            for barrier in rta['rta_barriers']:
                coefficients = barrier.get('control_coefficients')
                lower = barrier.get('lower_bound')
                if coefficients is not None and lower is not None:
                    residual = float(np.asarray(coefficients)@physical.as_array()-lower)
                    if residual < -1e-6:
                        excluded_by.append(dict(name=barrier['name'], kind=barrier['kind'],
                                                barrier_model=barrier['barrier_model'], residual=residual))
        _, _, terminated, truncated, info = env.step(executed)
        speed_before = float(np.linalg.norm(pre['velocity']))
        speed_after = float(np.linalg.norm(ego.velocity))
        yaw_rate = wrap_angle(ego.heading_theta-pre['heading'])/dt
        collided |= bool(info.get('crash_vehicle', False)) or bool(traffic.contact_roles)
        out |= bool(info.get('out_of_road', False))
        min_road_gap = min(min_road_gap, body_road_gap(ego, traffic.normal))
        trace.append(dict(step=index, time=(index+1)*dt, before=pre,
                          after=vehicle_snapshot(ego), nominal_action=nominal.tolist(),
                          witness_action=witness.tolist(), executed_action=np.asarray(executed).tolist(),
                          actual_acceleration=(speed_after-speed_before)/dt, actual_yaw_rate=yaw_rate,
                          actual_curvature=yaw_rate/max(.5*(speed_before+speed_after), .1),
                          road_gap=body_road_gap(ego, traffic.normal),
                          contact_roles=sorted(traffic.contact_roles),
                          verified_witness_action_excluded_by=excluded_by, rta=rta))
        if terminated or truncated or collided or out:
            break
    return dict(arm=arm, initial=initial, collision=collided,
                plugin=plugin.describe() if plugin is not None else None,
                rear_contact='rear' in traffic.contact_roles, front_contact='front' in traffic.contact_roles,
                out_of_road=out, min_body_road_gap=min_road_gap, steps=len(trace),
                completed_horizon=len(trace)==int(round(case.duration/dt)),
                final_lateral=float((np.asarray(env.agent.position)[:2]-traffic.origin)@traffic.normal),
                safe_witness=arm.startswith('witness') and not collided and not out and min_road_gap>0.
                             and len(trace)==int(round(case.duration/dt)),
                blocked_witness_steps=sum(bool(row['verified_witness_action_excluded_by']) for row in trace),
                infeasible_steps=sum(row['rta'] is not None and not row['rta']['rta_hard_feasible'] for row in trace),
                trace=trace)


def calibrate(env, case):
    adapter = MetaDriveCBFPlugin(CBFConfig())
    records = []
    for action in [[.15, 0.], [-.15, 0.], [0., 1.], [0., -.5]]:
        initialise(env, case)
        requested, _ = adapter._action_to_control(np.asarray(action), env.agent)
        before = vehicle_snapshot(env.agent)
        for _ in range(5):
            env.step(action)
        after = vehicle_snapshot(env.agent)
        mean_speed = .5*(np.linalg.norm(before['velocity'])+np.linalg.norm(after['velocity']))
        actual_yaw = wrap_angle(after['heading']-before['heading'])/.5
        records.append(dict(action=action, mapped_control=requested.as_array().tolist(),
                            actual_acceleration=float((np.linalg.norm(after['velocity'])-
                                                       np.linalg.norm(before['velocity']))/.5),
                            actual_curvature=float(actual_yaw/max(mean_speed, .1)),
                            approximate_wheelbase=adapter._wheelbase(env.agent),
                            actual_wheelbase=before['actual_wheelbase']))
    return records


def main():
    from eval_td3_policy_rta import build_rta_plugin
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--speed', type=float, default=10.)
    parser.add_argument('--rear_speed', type=float, default=15.)
    parser.add_argument('--rear_distance', type=float, default=12.)
    parser.add_argument('--front_distance', type=float, default=14.)
    parser.add_argument('--duration', type=float, default=6.)
    parser.add_argument('--rta_config', default='cfgs/rta_rear_escape_default.json')
    parser.add_argument('--plugin', choices=['fsm_rear_escape_cbf_qp', 'fsm_rear_physical_guard'],
                        default='fsm_rear_escape_cbf_qp')
    parser.add_argument('--require_rta_safe', action='store_true',
                        help='Fail after saving if the positive control or RTA arm fails')
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    case = Case('controlled_rear', args.speed, args.rear_speed, args.rear_distance,
                args.front_distance, args.duration)
    env = build_env(case)
    records = []
    try:
        calibration = calibrate(env, case)
        variants = [('nominal', None), ('witness', None),
                    ('legacy', build_rta_plugin('fsm_hocbf_qp', 'cfgs/rta_fsm_hocbf_default.json')),
                    ('development', build_rta_plugin(args.plugin, args.rta_config)),
                    ('witness_shadow_legacy', build_rta_plugin('fsm_hocbf_qp', 'cfgs/rta_fsm_hocbf_default.json')),
                    ('witness_shadow_development', build_rta_plugin(args.plugin, args.rta_config))]
        for arm, plugin in variants:
            record = run_arm(env, case, arm, plugin)
            records.append(record)
            print(json.dumps({k:v for k,v in record.items() if k not in ['trace', 'initial', 'plugin']}, ensure_ascii=False), flush=True)
    finally:
        env.close()
    initial_identical = all(r['initial'] == records[0]['initial'] for r in records)
    witness_valid = (records[1]['safe_witness'] and records[0]['rear_contact'] and initial_identical
                     and records[0]['initial']['ego']['actual_wheel_friction'] > 0.)
    report = dict(case=asdict(case), diagnostic_only=True, valid_positive_control=witness_valid,
                  initial_conditions_identical=initial_identical,
                  calibration=calibration, runs=records,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  note='A safe witness is a finite-horizon feasibility example, not a reachable-set proof or a controller candidate.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print('valid_positive_control=', witness_valid, 'saved=', destination, flush=True)
    candidate = next(r for r in records if r['arm'] == 'development')
    if args.require_rta_safe and (not witness_valid or candidate['collision'] or candidate['out_of_road']
                                  or not candidate['completed_horizon'] or candidate['min_body_road_gap'] <= 0.):
        raise SystemExit('Positive control or RTA safety check failed; see the saved report.')


if __name__ == '__main__':
    main()
