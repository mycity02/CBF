"""Build a collision-conditioned challenge set using a frozen no-RTA TD3 only.

All candidate attempts and rejections are retained in the manifest. Selection
never runs RTA. The selected set is conditional on baseline failure and must not
be reported as the accident rate of the original traffic distribution.
"""
import argparse
import copy
import hashlib
import json
import logging
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from metadrive.envs.scenario_env import ScenarioOnlineEnv
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.scenario.scenario_description import ScenarioDescription

from bmt.rl_train.train.scgen_rear_attack_generator import RearAttackSCGEN_Generator
from bmt.runtime_assurance.metrics import reference_route_length
from eval_rear_attack_timing import RearAttackAuditEnv
from eval_td3_policy import load_td3_model


def run_nominal(env, model, scenario, audit_env=None):
    env.set_scenario(ScenarioDescription(copy.deepcopy(scenario)))
    runner = audit_env if audit_env is not None else env
    observation, info = runner.reset(seed=0)
    centered = copy.deepcopy(env.engine.data_manager.current_scenario)
    positions, headings, velocities = [], [], []

    def record():
        positions.append(np.asarray(env.agent.position, float)[:2].copy())
        headings.append(float(env.agent.heading_theta))
        velocities.append(np.asarray(env.agent.velocity, float)[:2].copy())

    record()
    steps = 0
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = runner.step(action)
        record()
        steps += 1
        done = bool(terminated or truncated)
    result = {
        'steps': steps,
        'route_length': reference_route_length(env),
        'collision': bool(info.get('crash', False)),
        'arrive_dest': bool(info.get('arrive_dest', False)),
        'out_of_road': bool(info.get('out_of_road', False)),
        'ego_length': float(env.agent.LENGTH),
        'ego_width': float(env.agent.WIDTH),
    }
    if audit_env is not None:
        result['rear_attack_audit'] = copy.deepcopy(audit_env.audit)
    reference = {'position': np.asarray(positions), 'heading': np.asarray(headings),
                 'velocity': np.asarray(velocities)}
    return result, reference, centered


def initial_quality(scenario, reference, ego_length):
    state = scenario['tracks'][scenario['metadata']['new_adv_id']]['state']
    if not bool(state['valid'][0]):
        return 'attacker_not_valid_at_start'
    heading = float(reference['heading'][0])
    forward = np.array([np.cos(heading), np.sin(heading)])
    normal = np.array([-np.sin(heading), np.cos(heading)])
    relative = state['position'][0, :2] - reference['position'][0]
    longitudinal, lateral = float(relative @ forward), float(relative @ normal)
    delta = float(state['heading'][0] - heading)
    heading_error = abs(float(np.arctan2(np.sin(delta), np.cos(delta))))
    gap = -longitudinal - .5 * (ego_length + float(state['length'][0]))
    if longitudinal > -8.1 or gap < 3.0:
        return 'insufficient_initial_rear_separation'
    if abs(lateral) > 3.0 or heading_error > np.pi / 4:
        return 'not_a_same_direction_rear_approach'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_dir', required=True)
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--candidate_dir', required=True)
    parser.add_argument('--selected_dir', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--num_sources', type=int, default=100)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--target_count', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--initial_mode_filter', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    destination, candidates, manifest_path = Path(args.selected_dir), Path(args.candidate_dir), Path(args.manifest)
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(manifest_path)
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for key in ['source_dir', 'ckpt_path', 'candidate_dir', 'selected_dir', 'seed', 'initial_mode_filter']:
            previous = manifest['arguments'].get(key, False) if key == 'initial_mode_filter' else manifest['arguments'][key]
            if previous != getattr(args, key):
                raise ValueError(f'Resume argument changed: {key}')
    else:
        if any(p.exists() and any(p.iterdir()) for p in [destination, candidates]):
            raise FileExistsError('Use new or empty candidate and selected directories')
        manifest = {'selection': 'frozen_no_rta_td3_confirmed_rear_collision',
                    'arguments': vars(args).copy(), 'sources': [], 'attempts': [], 'selected': [],
                    'checkpoint_sha256': hashlib.sha256(Path(args.ckpt_path).read_bytes()).hexdigest(),
                    'notes': ['RTA is never evaluated during selection.',
                              'This set is conditioned on baseline rear-collision failure.',
                              'Source route, background traffic, native rewards and terminations are preserved.',
                              'Early attacker trajectories use BMT before anchor and constant velocity afterwards.']}
    destination.mkdir(parents=True, exist_ok=True)
    candidates.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    model = load_td3_model(args.ckpt_path, 'cpu')
    generator = RearAttackSCGEN_Generator(rear_offset=3.8, initial_rear_filter=args.initial_mode_filter)
    env = ScenarioOnlineEnv(dict(
        store_map=False, use_render=False, manual_control=False, show_interface=False,
        agent_policy=EnvInputPolicy, num_scenarios=1, start_scenario_index=0,
        horizon=100, reactive_traffic=False, no_static_vehicles=True, no_light=True,
        crash_vehicle_done=True, out_of_route_done=False, crash_object_done=False,
        crash_human_done=False, relax_out_of_road_done=False, log_level=logging.CRITICAL,
    ))
    audit_env = RearAttackAuditEnv(env)
    files = sorted(Path(args.source_dir).glob('sd_*.pkl'))
    completed = {s['source_index'] for s in manifest['sources']}
    known_attempts = {a['filename']: a for a in manifest['attempts']}

    def save():
        temporary = manifest_path.with_suffix(manifest_path.suffix + '.tmp')
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        temporary.replace(manifest_path)

    save()
    try:
        for source_index in range(args.start_index, min(len(files), args.start_index + args.num_sources)):
            if len(manifest['selected']) >= args.target_count:
                break
            if source_index in completed:
                continue
            with files[source_index].open('rb') as stream:
                source = pickle.load(stream)
            baseline, reference, centered = run_nominal(env, model, source)
            source_row = {'source_index': source_index, 'scenario_id': source['id'],
                          'native_baseline': baseline}
            length = baseline['route_length']
            anchor = min(30, ((len(reference['position']) - 6) // 5) * 5)
            if length is None or length < 2.0 or anchor < 20:
                source_row['status'] = 'insufficient_valid_nominal_time_window'
                manifest['sources'].append(source_row)
                save()
                continue
            source_row['status'] = 'tested'
            for variant, speed_delta in enumerate([3.0, 6.0]):
                seed = args.seed + source_index * 100 + variant
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                generator.attack_time = anchor * .1
                generator.speed_delta = speed_delta
                filename = f"sd_{source['id']}_td3_rear_t{anchor}_dv{int(speed_delta)}.pkl"
                previous_attempt = known_attempts.get(filename)
                if previous_attempt is not None:
                    if previous_attempt['status'] == 'accepted_rear_collision':
                        break
                    continue
                candidate_path = candidates / filename
                if candidate_path.exists():
                    with candidate_path.open('rb') as stream:
                        candidate = pickle.load(stream)
                    if candidate['metadata']['rear_attack']['generation_seed'] != seed:
                        raise ValueError('Cached candidate seed does not match')
                else:
                    candidate = generator.generate_from_reference_rollout(centered, reference)
                    candidate['metadata']['rear_attack'].update({
                        'source_index': source_index, 'generation_seed': seed,
                        'baseline_checkpoint_sha256': manifest['checkpoint_sha256'],
                    })
                    with candidate_path.open('xb') as stream:
                        pickle.dump(candidate, stream)
                attempt = {'source_index': source_index, 'scenario_id': source['id'],
                           'filename': filename, 'anchor_step': anchor,
                           'speed_delta': speed_delta, 'seed': seed}
                rejection = initial_quality(candidate, reference, baseline['ego_length'])
                if rejection:
                    attempt['status'] = rejection
                else:
                    result, _, _ = run_nominal(env, model, candidate, audit_env)
                    audit = result['rear_attack_audit']
                    attempt['evaluation'] = result
                    contact_step = audit['first_attacker_contact_step']
                    if audit['rear_collision'] and contact_step is not None and contact_step >= 10:
                        attempt['status'] = 'accepted_rear_collision'
                        selected_path = destination / filename
                        if not selected_path.exists():
                            selected_path.write_bytes(candidate_path.read_bytes())
                        elif selected_path.read_bytes() != candidate_path.read_bytes():
                            raise ValueError('Existing selected scenario differs from candidate')
                        manifest['selected'].append({'filename': filename, 'source_index': source_index,
                                                     'scenario_id': source['id'], 'candidate_seed': seed})
                    else:
                        attempt['status'] = 'no_confirmed_rear_collision'
                manifest['attempts'].append(attempt)
                save()
                print(f"source={source_index} variant={variant} status={attempt['status']} selected={len(manifest['selected'])}", flush=True)
                if attempt['status'] == 'accepted_rear_collision':
                    break
            manifest['sources'].append(source_row)
            save()
    finally:
        env.close()
    print(json.dumps({'sources': len(manifest['sources']), 'candidates': len(manifest['attempts']),
                      'selected': len(manifest['selected']), 'target': args.target_count}, indent=2))


if __name__ == '__main__':
    main()
