"""Run with python -m unittest discover -s tests -p test_rear_attack_timing.py."""
import unittest

import numpy as np
import torch

from bmt.rl_train.train.scgen_rear_attack_generator import (
    create_rear_attack_adv_batched,
    extend_post_anchor_track,
    resolve_attack_step,
    RearAttackSCGEN_Generator,
)


def batch():
    n = 91
    positions = torch.zeros(1, n, 1, 3)
    positions[0, :, 0, 0] = torch.arange(n)
    return {
        "decoder/sdc_index": torch.tensor([0]),
        "decoder/agent_position": positions,
        "decoder/agent_heading": torch.zeros(1, n, 1),
        "decoder/agent_velocity": torch.ones(1, n, 1, 2),
        "decoder/agent_shape": torch.full((1, n, 1, 3), 4.0),
        "decoder/agent_valid_mask": torch.ones(1, n, 1, dtype=torch.bool),
        "decoder/current_agent_shape": torch.ones(1, 1, 3),
        "decoder/current_agent_position": torch.zeros(1, 1, 3),
        "decoder/agent_type": torch.ones(1, 1, dtype=torch.long),
        "decoder/agent_id": torch.tensor([[0]]),
        "decoder/object_of_interest_id": torch.tensor([[0]]),
        "decoder/current_agent_valid_mask": torch.ones(1, 1, dtype=torch.bool),
    }


class RearAttackTimingTests(unittest.TestCase):
    def test_early_anchor_preserves_source_and_truncates_only_attacker(self):
        data, idx, steps = create_rear_attack_adv_batched(batch(), attack_step=30)
        self.assertEqual(steps.tolist(), [30])
        self.assertEqual(int(data["decoder/agent_valid_mask"][0, :, idx].sum()), 31)
        self.assertTrue(bool(data["decoder/agent_valid_mask"][0, :, 0].all()))
        self.assertAlmostEqual(float(data["decoder/agent_position"][0, 30, idx, 0]), 26.6, places=4)
        self.assertEqual(float(data["decoder/agent_position"][0, 30, 0, 0]), 30.0)

    def test_default_keeps_final_anchor(self):
        _, _, steps = create_rear_attack_adv_batched(batch())
        self.assertEqual(steps.tolist(), [90])

    def test_rejects_invalid_ego_anchor(self):
        data = batch()
        data["decoder/agent_valid_mask"][0, 30, 0] = False
        with self.assertRaises(ValueError):
            create_rear_attack_adv_batched(data, attack_step=30)

    def test_continuation_uses_replay_clock_despite_irregular_source_timestamps(self):
        state = {"position": np.array([[1., 2.], [2., 2.], [0., 0.], [0., 0.]]),
                 "velocity": np.array([[1., 0.], [2., 1.], [0., 0.], [0., 0.]]),
                 "heading": np.array([0., .2, 0., 0.]),
                 "valid": np.array([True, True, False, False])}
        scenario = {"metadata": {"new_adv_id": "99999", "ts": [0., .1, .2, .4]},
                    "tracks": {"99999": {"state": state}}}
        extend_post_anchor_track(scenario, 1)
        np.testing.assert_allclose(state["position"], [[1., 2.], [2., 2.], [2.2, 2.1], [2.4, 2.2]])
        self.assertTrue(state["valid"].all())
        np.testing.assert_allclose(state["heading"][2:], .2)

    def test_attack_time_matches_fixed_token_and_replay_clocks(self):
        self.assertEqual(resolve_attack_step(3.0, .1, 5, 91), 30)
        self.assertEqual(resolve_attack_step(9.0, .1, 5, 91), 90)
        for invalid in [0.0, 3.1, 9.5]:
            with self.assertRaises(ValueError):
                resolve_attack_step(invalid, .1, 5, 91)

    def test_optional_mode_filter_rejects_initially_close_attacker(self):
        generator = RearAttackSCGEN_Generator.__new__(RearAttackSCGEN_Generator)
        generator.num_modes = 2
        generator.initial_rear_filter = True
        generator.reject_multi_batch = lambda output, count: (0, 0.0)
        original = torch.zeros(2, 91, 2, 3)
        original[:, -1, 0, 0] = 30.0
        reconstructed = original[..., :2].clone()
        reconstructed[:, -1, 1, 0] = 26.2
        reconstructed[0, 0, 1, 0] = -4.0
        reconstructed[1, 0, 1, 0] = -15.0
        output = {'decoder/reconstructed_valid_mask': torch.ones(2, 91, 2, dtype=torch.bool),
                  'decoder/sdc_index': torch.zeros(2, dtype=torch.long),
                  'decoder/agent_position': original,
                  'decoder/agent_heading': torch.zeros(2, 91, 2),
                  'decoder/reconstructed_position': reconstructed,
                  'decoder/reconstructed_heading': torch.zeros(2, 91, 2)}
        self.assertEqual(generator._select_rear_attack_mode(output, 1), 1)
        generator.initial_rear_filter = False
        self.assertEqual(generator._select_rear_attack_mode(output, 1), 0)


if __name__ == "__main__":
    unittest.main()
