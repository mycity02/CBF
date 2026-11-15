"""Compare complete, fixed-cohort RTA speed trials without changing metrics."""
import argparse
import json
from pathlib import Path


def read_report(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def check_pair(reference, trial):
    a, b = reference['reproducibility'], trial['reproducibility']
    checks = {
        'checkpoint_identical': a['checkpoint_sha256'] == b['checkpoint_sha256'],
        'scenario_files_identical': a['scenario_sha256'] == b['scenario_sha256'],
        'scenario_files_unchanged': bool(b['scenario_hashes_unchanged']),
        'episode_indices_identical': [e['scenario_index'] for e in reference['episodes']]
                                     == [e['scenario_index'] for e in trial['episodes']],
        'degenerate_count_identical': reference['summary']['degenerate_scenario_count']
                                      == trial['summary']['degenerate_scenario_count'],
        'control_dt_identical': abs(reference['summary']['control_step_seconds']
                                    - trial['summary']['control_step_seconds']) < 1e-9,
    }
    if not all(checks.values()):
        raise ValueError(f'Incompatible paired evaluation: {checks}')
    return checks


def pack(report):
    s = report['summary']
    es = report['episodes']
    return {
        'num_episodes': len(es),
        'collisions': sum(bool(e['collision']) for e in es),
        'road_events': sum(bool(e['out_of_road']) for e in es),
        'clean_arrivals': sum(bool(e['arrive_dest'] and not e['collision'] and not e['out_of_road']) for e in es),
        'arrivals': sum(bool(e['arrive_dest']) for e in es),
        'termination_counts': s['termination_counts'],
        'confirmed_rear_collisions': s.get('rear_attack_coverage', {}).get('rear_collision_episodes'),
        'reward': s['reward'], 'completion': s['completion'],
        'avg_speed_mps': s['avg_speed_mps'],
    }


def safety_gates(reference, trial):
    a, b = pack(reference), pack(trial)
    result = {
        'no_more_collisions': b['collisions'] <= a['collisions'],
        'no_more_road_events': b['road_events'] <= a['road_events'],
        'no_fewer_clean_arrivals': b['clean_arrivals'] >= a['clean_arrivals'],
    }
    if a['confirmed_rear_collisions'] is not None and b['confirmed_rear_collisions'] is not None:
        result['no_more_confirmed_rear_collisions'] = b['confirmed_rear_collisions'] <= a['confirmed_rear_collisions']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='analysis_outputs/rta_diagnostics')
    parser.add_argument('--before_100', default='rear_guard_v16_aligned_100.json',
                        help='Reference report filename within --directory')
    parser.add_argument('--before_18', default='rear_guard_v16_aligned_18.json',
                        help='Reference report filename within --directory')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(args.directory)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(output)
    no_rta = read_report(root/'speed5_none_refresh_100.json')
    old100 = read_report(root/args.before_100)
    old18 = read_report(root/args.before_18)
    check_pair(old100, no_rta)
    nominal_speed = no_rta['summary']['avg_speed_mps']
    trials = []
    for path in sorted(root.glob('speed5_v*_100.json')):
        report = read_report(path)
        info = dict(report=path.name, metrics_100=pack(report), checks_100=check_pair(old100, report))
        info['speed_reduction_percent'] = 100.*(1.-info['metrics_100']['avg_speed_mps']/nominal_speed)
        info['five_percent_speed_met'] = info['speed_reduction_percent'] <= 5.
        info['seven_percent_speed_met'] = info['speed_reduction_percent'] <= 7.
        info['gates_100'] = safety_gates(old100, report)
        challenge_path = path.with_name(path.name.replace('_100.json', '_18.json'))
        if challenge_path.exists():
            challenge = read_report(challenge_path)
            info.update(report_18=challenge_path.name, metrics_18=pack(challenge),
                        checks_18=check_pair(old18, challenge), gates_18=safety_gates(old18, challenge))
            info['same_controller_source_in_both_runs'] = (
                report['reproducibility']['controller_source_sha256']
                == challenge['reproducibility']['controller_source_sha256'])
        else:
            info['metrics_18'] = None
        info['safety_no_worse_on_both_sets'] = (info['metrics_18'] is not None
            and all(info['gates_100'].values()) and all(info['gates_18'].values())
            and info.get('same_controller_source_in_both_runs', False))
        trials.append(info)
    achieved = [t for t in trials if t['five_percent_speed_met'] and t['safety_no_worse_on_both_sets']]
    result = {
        'status': 'five_percent_with_safety_met' if achieved else 'five_percent_with_safety_not_met',
        'metric_policy': 'unchanged original avg_speed_mps over all 100 scenarios; no filtering or reweighting',
        'no_rta_refreshed': pack(no_rta), 'before_100': pack(old100), 'before_18': pack(old18),
        'reference_reports': {'100': args.before_100, '18': args.before_18},
        'speed_target_5_mps': .95*nominal_speed, 'speed_target_7_mps': .93*nominal_speed,
        'default_configuration_changed': False, 'trials': trials,
        'promotion_policy': (
            'Fixed-cohort gates alone are insufficient: native known-avoidable rear positive '
            'controls must also pass. This summary does not automatically promote any configuration.'),
        'native_positive_controls_included_in_gates': False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'status': result['status'], 'trials': len(trials),
                      'output': str(output)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
