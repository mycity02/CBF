from copy import deepcopy

import pytest

from summarize_rta_efficiency import check_pair, pack, safety_gates


def report():
    return {
        'reproducibility': {'checkpoint_sha256': 'model', 'scenario_sha256': {'scene': 'hash'},
                            'scenario_hashes_unchanged': True},
        'summary': {'degenerate_scenario_count': 0, 'control_step_seconds': .1,
                    'termination_counts': {'success': 1, 'collision': 1},
                    'rear_attack_coverage': {'rear_collision_episodes': 1},
                    'reward': 1., 'completion': .5, 'avg_speed_mps': 10.},
        'episodes': [dict(scenario_index=0, collision=False, out_of_road=False, arrive_dest=True),
                     dict(scenario_index=1, collision=True, out_of_road=False, arrive_dest=True)],
    }


def test_collision_plus_arrival_is_not_counted_as_clean_success():
    metrics = pack(report())
    assert metrics['arrivals'] == 2
    assert metrics['clean_arrivals'] == 1
    assert metrics['collisions'] == 1


def test_changed_scene_files_are_rejected_even_if_summary_is_better():
    a = report()
    b = deepcopy(a)
    assert all(check_pair(a, b).values())
    b['reproducibility']['scenario_sha256']['scene'] = 'different'
    with pytest.raises(ValueError, match='Incompatible'):
        check_pair(a, b)


def test_safety_gate_keeps_rear_collisions_and_road_events_visible():
    a = report()
    b = deepcopy(a)
    b['summary']['rear_attack_coverage']['rear_collision_episodes'] = 2
    b['episodes'][0]['out_of_road'] = True
    gates = safety_gates(a, b)
    assert gates['no_more_collisions']
    assert not gates['no_more_road_events']
    assert not gates['no_fewer_clean_arrivals']
    assert not gates['no_more_confirmed_rear_collisions']
