"""Offline diagnostic only: execute FSM references WITHOUT the safety QP.

These results are not an RTA method or a safety guarantee. They separate
reference-generation failures from failures introduced by QP constraints.
No scene IDs, attacker metadata, future states, or action rollouts are used.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bmt.runtime_assurance.types import Control2D
from eval_rear_attack_timing import RearAttackAuditEnv
from eval_td3_policy import build_eval_env, load_td3_model
from eval_td3_policy_rta import build_rta_plugin, evaluate


class ReferenceDiagnostic:
    def __init__(self, config):
        self.backend = build_rta_plugin('fsm_rear_escape_cbf_qp', config)

    def reset(self, env):
        self.backend.reset(env)

    def describe(self):
        return dict(name='UNFILTERED_REFERENCE_DIAGNOSTIC', not_rta=True,
                    backend=self.backend.describe())

    def filter_action(self, env, nominal):
        _, telemetry = self.backend.filter_action(env, nominal)
        cfg = self.backend.filter.config
        control = Control2D(float(np.clip(telemetry['rta_reference_acceleration'],
                                         cfg.min_acceleration, cfg.max_acceleration)),
                            float(np.clip(telemetry['rta_reference_curvature'],
                                          -cfg.max_curvature, cfg.max_curvature)))
        ego = env.agent
        _, limit = self.backend._action_to_control(nominal, ego)
        action = self.backend._control_to_action(control, ego, limit)
        telemetry.update(rta_plugin='UNFILTERED_REFERENCE_DIAGNOSTIC',
                         diagnostic_no_safety_projection=True,
                         rta_safe_action=action.tolist(),
                         rta_safe_acceleration=control.acceleration,
                         rta_safe_curvature=control.curvature)
        return action, telemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--eval_data_dir', required=True)
    parser.add_argument('--rta_config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--num_episodes', type=int, default=18)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(output)
    model = load_td3_model(args.ckpt_path, 'cpu')
    env, count = build_eval_env(Path(args.eval_data_dir).resolve(), 100)
    wrapped = RearAttackAuditEnv(env, 3.)
    plugin = ReferenceDiagnostic(args.rta_config)
    try:
        summary, episodes = evaluate(model, wrapped, args.num_episodes, 0, count, True, plugin)
        for episode, audit in zip(episodes, wrapped.audits):
            episode['rear_attack_audit'] = audit
        result = dict(diagnostic_only=True, not_rta=True, summary=summary, episodes=episodes,
                      controller=plugin.describe())
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(summary['termination_counts']))
    finally:
        env.close()


if __name__ == '__main__':
    main()
