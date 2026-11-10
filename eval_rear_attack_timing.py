"""Paired timing smoke tests with attacker-ID and rear-approach telemetry."""
import argparse
import json
from pathlib import Path

import numpy as np

from eval_td3_policy import build_eval_env, load_td3_model
from eval_td3_policy_rta import build_rta_plugin, evaluate


def is_rear_impact_point(longitudinal, lateral, length, width):
    """Rear quarter of the ego body, with a small numerical edge tolerance."""
    return longitudinal <= -.35 * length and abs(lateral) <= .5 * width + .2


class RearAttackAuditEnv:
    def __init__(self, env, ttc_threshold=3.0):
        self.env = env
        self.ttc_threshold = ttc_threshold
        self.audits = []
        self.audit = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        scenario = self.env.engine.data_manager.current_scenario
        metadata = scenario['metadata']
        timing = metadata.get('rear_attack', {})
        attack_id = str(metadata.get('new_adv_id', '99999'))
        anchor = timing.get('anchor_step')
        if anchor is None and attack_id in scenario['tracks']:
            common = np.asarray(scenario['tracks'][attack_id]['state']['valid'], bool).copy()
            common &= np.asarray(scenario['tracks'][str(metadata['sdc_id'])]['state']['valid'], bool)
            valid = np.flatnonzero(common)
            anchor = int(valid[-1]) if len(valid) else None
        self.audit = {
            'scenario_id': scenario['id'], 'attacker_id': attack_id,
            'anchor_step': anchor, 'attacker_observed': False,
            'rear_approach_observed': False, 'rear_threat_observed': False,
            'rear_threat_at_reset': False, 'first_rear_threat_step': None,
            'first_attacker_contact_step': None,
            'attacker_contact': False, 'rear_attacker_contact': False,
            'rear_collision': False,
            'min_rear_gap_m': None, 'min_rear_ttc_seconds': None,
            'trace': [],
        }
        self._snapshot(check_contact=False)
        return result

    def step(self, action):
        result = self.env.step(action)
        self._snapshot(check_contact=True)
        if result[2] or result[3]:
            self.audit['last_step'] = int(self.env.episode_step)
            anchor = self.audit['anchor_step']
            self.audit['reached_anchor'] = anchor is not None and self.audit['last_step'] >= anchor
            self.audits.append(self.audit)
        return result

    def _snapshot(self, check_contact):
        manager = self.env.engine.traffic_manager
        mapping = manager._scenario_id_to_obj_id
        obj_id = mapping.get(self.audit['attacker_id'])
        attacker = manager.spawned_objects.get(obj_id)
        if attacker is None:
            return
        self.audit['attacker_observed'] = True
        ego = self.env.agent
        heading = np.array([np.cos(ego.heading_theta), np.sin(ego.heading_theta)])
        normal = np.array([-heading[1], heading[0]])
        relative = np.asarray(attacker.position)[:2] - np.asarray(ego.position)[:2]
        longitudinal = float(relative @ heading)
        lateral = float(relative @ normal)
        relative_velocity = np.asarray(attacker.velocity)[:2] - np.asarray(ego.velocity)[:2]
        closing = float(relative_velocity @ heading)
        delta_heading = float(attacker.heading_theta - ego.heading_theta)
        long_extent = float(.5 * (ego.LENGTH + attacker.LENGTH * abs(np.cos(delta_heading)) + attacker.WIDTH * abs(np.sin(delta_heading))))
        lat_extent = float(.5 * (ego.WIDTH + attacker.LENGTH * abs(np.sin(delta_heading)) + attacker.WIDTH * abs(np.cos(delta_heading))))
        in_rear_corridor = longitudinal < 0 and abs(lateral) <= lat_extent
        gap = -longitudinal - long_extent
        ttc = max(0.0, gap) / closing if in_rear_corridor and closing > .1 else None
        if in_rear_corridor:
            previous = self.audit['min_rear_gap_m']
            self.audit['min_rear_gap_m'] = gap if previous is None else min(previous, gap)
        if ttc is not None:
            self.audit['rear_approach_observed'] = True
            previous = self.audit['min_rear_ttc_seconds']
            self.audit['min_rear_ttc_seconds'] = ttc if previous is None else min(previous, ttc)
            self.audit['rear_threat_observed'] |= ttc <= self.ttc_threshold
            if ttc <= self.ttc_threshold and self.audit['first_rear_threat_step'] is None:
                self.audit['first_rear_threat_step'] = int(self.env.episode_step)
                self.audit['rear_threat_at_reset'] = self.env.episode_step == 0
        contact = False
        rear_impact = False
        contact_points = []
        if check_contact:
            contacts = self.env.engine.physics_world.dynamic_world.contactTestPair(ego.chassis.node(), attacker.chassis.node())
            for item in contacts.getContacts():
                point = item.getManifoldPoint()
                if point.getDistance() > 0:
                    continue
                contact = True
                world_point = point.getPositionWorldOnA() if item.getNode0() == ego.chassis.node() else point.getPositionWorldOnB()
                relative_point = np.array([world_point[0], world_point[1]]) - np.asarray(ego.position)[:2]
                point_long, point_lat = float(relative_point @ heading), float(relative_point @ normal)
                contact_points.append([point_long, point_lat])
                rear_impact |= is_rear_impact_point(point_long, point_lat, ego.LENGTH, ego.WIDTH)
            self.audit['attacker_contact'] |= contact
            self.audit['rear_attacker_contact'] |= contact and longitudinal < 0
            aligned = abs(np.arctan2(np.sin(delta_heading), np.cos(delta_heading))) <= np.pi / 4
            confirmed = bool(contact and longitudinal < 0 and rear_impact and aligned and ego.crash_vehicle)
            self.audit['rear_collision'] |= confirmed
            if contact and self.audit['first_attacker_contact_step'] is None:
                self.audit['first_attacker_contact_step'] = int(self.env.episode_step)
        self.audit['trace'].append({
            'step': int(self.env.episode_step), 'longitudinal_m': longitudinal,
            'lateral_m': lateral, 'closing_speed_mps': closing,
            'rear_gap_m': gap if in_rear_corridor else None,
            'rear_ttc_seconds': ttc, 'attacker_contact': contact,
            'ego_contact_points_local_m': contact_points,
            'native_vehicle_collision': bool(ego.crash_vehicle),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval_data_dir', required=True)
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--rta_plugin', default='none', choices=['none', 'fsm_hocbf_qp'])
    parser.add_argument('--rta_config', default=None)
    parser.add_argument('--ttc_threshold', type=float, default=3.0)
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing report: {destination}')
    model = load_td3_model(args.ckpt_path, args.device)
    env, count = build_eval_env(Path(args.eval_data_dir).resolve(), args.horizon, args.start_index)
    wrapped = RearAttackAuditEnv(env, args.ttc_threshold)
    plugin = build_rta_plugin(args.rta_plugin, args.rta_config)
    try:
        summary, episodes = evaluate(model, wrapped, args.num_episodes, args.start_index, count, True, plugin)
        for episode, audit in zip(episodes, wrapped.audits):
            episode['rear_attack_audit'] = audit
        summary['rear_attack_coverage'] = {
            'ttc_threshold_seconds': args.ttc_threshold,
            **{key + '_episodes': sum(bool(a[key]) for a in wrapped.audits)
               for key in ['attacker_observed', 'rear_approach_observed', 'rear_threat_observed',
                           'rear_threat_at_reset', 'reached_anchor', 'attacker_contact', 'rear_attacker_contact',
                           'rear_collision']},
        }
        report = {'checkpoint': str(Path(args.ckpt_path).resolve()),
                  'eval_data_dir': str(Path(args.eval_data_dir).resolve()),
                  'runtime_assurance': {'plugin': args.rta_plugin, 'config': args.rta_config},
                  'summary': summary, 'episodes': episodes}
        serialized = json.dumps(report, indent=2, ensure_ascii=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x', encoding='utf-8') as stream:
            stream.write(serialized)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        env.close()


if __name__ == '__main__':
    main()
