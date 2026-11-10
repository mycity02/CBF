import unittest

from eval_rear_attack_timing import is_rear_impact_point


class RearImpactTests(unittest.TestCase):
    def test_rear_bumper_is_included(self):
        self.assertTrue(is_rear_impact_point(-2.2, .3, 4.5, 1.8))

    def test_front_and_middle_side_are_excluded(self):
        self.assertFalse(is_rear_impact_point(2.2, .1, 4.5, 1.8))
        self.assertFalse(is_rear_impact_point(-.5, .9, 4.5, 1.8))

    def test_far_lateral_point_is_excluded(self):
        self.assertFalse(is_rear_impact_point(-2.2, 2.0, 4.5, 1.8))


if __name__ == '__main__':
    unittest.main()
