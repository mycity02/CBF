from types import SimpleNamespace

import numpy as np

from cbf_rta.metrics import (
    control_step_seconds,
    latency_stats,
    motion_summary,
    speed_stats,
)


def test_control_step_seconds_uses_metadrive_physics_configuration():
    env = SimpleNamespace(
        config={
            "physics_world_step_size": 0.02,
            "decision_repeat": 5,
        }
    )

    assert control_step_seconds(env) == 0.1


def test_latency_stats_reports_percentiles_and_deadline_misses():
    stats = latency_stats([1.0, 2.0, 200.0], deadline_ms=100.0)

    assert stats["rta_filter_calls"] == 3
    np.testing.assert_allclose(stats["rta_filter_mean_ms"], 203.0 / 3.0)
    np.testing.assert_allclose(stats["rta_filter_p95_ms"], 180.2)
    np.testing.assert_allclose(stats["rta_deadline_miss_rate"], 1.0 / 3.0)
    np.testing.assert_allclose(stats["rta_budget_utilization"], 203.0 / 300.0)


def test_motion_summary_uses_all_steps_and_arrival_subset():
    episodes = [
        {
            "episode_duration_seconds": 2.0,
            "distance_m": 8.0,
            "arrive_dest": True,
            "avg_speed_mps": 4.0,
        },
        {
            "episode_duration_seconds": 3.0,
            "distance_m": 6.0,
            "arrive_dest": False,
            "avg_speed_mps": 2.0,
        },
    ]

    summary = motion_summary(episodes, speed_samples_mps=[0.0, 2.0, 4.0, 6.0])

    assert summary["arrived_episode_count"] == 1
    assert summary["arrived_avg_time_seconds"] == 2.0
    assert summary["arrived_avg_speed_mps"] == 4.0
    np.testing.assert_allclose(summary["avg_distance_speed_mps"], 14.0 / 5.0)
    np.testing.assert_allclose(summary["avg_speed_mps"], 3.0)
    assert speed_stats([0.0, 2.0])["low_speed_rate"] == 0.5
