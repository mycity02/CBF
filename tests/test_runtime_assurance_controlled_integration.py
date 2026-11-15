"""Opt-in native physics regression; ordinary unit tests need no MetaDrive."""
import json
import os
from pathlib import Path
import subprocess
import uuid

import pytest


@pytest.mark.skipif(not os.environ.get('RTA_NATIVE_TEST_PYTHON'),
                    reason='Set RTA_NATIVE_TEST_PYTHON to the adv-bmt Python to run native physics checks')
@pytest.mark.parametrize('speed,rear_speed', [(.5, 5.5), (3., 8.), (10., 15.)])
@pytest.mark.parametrize('config', ['cfgs/rta_rear_physical.json', 'cfgs/rta_rear_physical_v4.json',
                                   'cfgs/rta_rear_physical_v15_rejoin.json',
                                   'cfgs/rta_rear_physical_v16_aligned.json',
                                   'cfgs/rta_efficiency_v5_native_drive.json',
                                   'cfgs/rta_efficiency_v12_low_speed_domain.json',
                                   'cfgs/rta_efficiency_v13_native_w1.json'])
def test_known_avoidable_rear_case_passes_with_real_vehicle_physics(speed, rear_speed, config, request):
    if speed == .5 and config in ('cfgs/rta_efficiency_v5_native_drive.json',
                                  'cfgs/rta_efficiency_v12_low_speed_domain.json'):
        # These experiments were rejected after real rear impacts. Keep running
        # the original strict assertions as expected regressions, not weakened
        # safety gates. An unexpected pass must prompt a fresh promotion audit.
        request.applymarker(pytest.mark.xfail(strict=True,
            reason='Rejected efficiency prototype: known low-speed rear collision; do not deploy'))
    root = Path(__file__).resolve().parents[1]
    # pytest's private Windows temp-directory ACLs can exclude a sandboxed
    # Python process. Use a unique, non-overwriting report in the workspace
    # instead; retain it as reproducible integration-test evidence.
    report_directory = root/'analysis_outputs/rta_diagnostics'
    report_directory.mkdir(parents=True, exist_ok=True)
    output = report_directory/f'native_integration_{speed}_{uuid.uuid4().hex}.json'
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = subprocess.run([
        os.environ['RTA_NATIVE_TEST_PYTHON'], '-u', 'diagnose_rta_controlled_rear.py',
        '--speed', str(speed), '--rear_speed', str(rear_speed),
        '--plugin', 'fsm_rear_physical_guard', '--rta_config', config,
        '--require_rta_safe', '--out', str(output),
    ], cwd=root, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['initial_conditions_identical']
    assert report['valid_positive_control']
    nominal = next(r for r in report['runs'] if r['arm'] == 'nominal')
    candidate = next(r for r in report['runs'] if r['arm'] == 'development')
    assert nominal['rear_contact']
    assert not candidate['collision'] and not candidate['out_of_road']
    assert candidate['completed_horizon']
    assert candidate['min_body_road_gap'] > 0.
