from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


LOW_SPEED_THRESHOLD_MPS = 1.0
STOPPED_THRESHOLD_MPS = 0.1


def termination_outcome(
    arrive_dest: bool,
    out_of_road: bool,
    collision: bool,
) -> str:
    """Return an exclusive outcome using MetaDrive ScenarioEnv precedence."""

    if arrive_dest:
        return "success"
    if out_of_road:
        return "road_violation"
    if collision:
        return "collision"
    return "timeout_or_other"


def reference_route_length(env: Any) -> Optional[float]:
    ego = getattr(env, "agent", None)
    navigation = getattr(ego, "navigation", None)
    trajectory = getattr(navigation, "reference_trajectory", None)
    value = getattr(trajectory, "length", None)
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def termination_summary(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize exclusive outcomes and keep raw event flags separate."""

    total = len(episodes)
    counts = {
        "success": 0,
        "road_violation": 0,
        "collision": 0,
        "timeout_or_other": 0,
    }
    for episode in episodes:
        outcome = str(episode["termination_reason"])
        counts[outcome] = counts.get(outcome, 0) + 1

    valid = [episode for episode in episodes if not episode["degenerate_scenario"]]

    def rate(count: int, denominator: int = total) -> float:
        return float(count) / denominator if denominator else 0.0

    def valid_mean(key: str) -> float:
        values = np.asarray(
            [float(episode.get(key, 0.0)) for episode in valid],
            dtype=float,
        )
        return float(values.mean()) if len(values) else 0.0

    valid_road_events = sum(bool(episode["out_of_road"]) for episode in valid)
    valid_road_terminations = sum(
        episode["termination_reason"] == "road_violation" for episode in valid
    )
    degenerate_count = total - len(valid)
    return {
        "termination_counts": counts,
        "success_termination_rate": rate(counts["success"]),
        "road_violation_termination_rate": rate(counts["road_violation"]),
        "collision_termination_rate": rate(counts["collision"]),
        "timeout_or_other_rate": rate(counts["timeout_or_other"]),
        "degenerate_scenario_count": degenerate_count,
        "degenerate_scenario_rate": rate(degenerate_count),
        "valid_dynamic_scenario_count": len(valid),
        "valid_dynamic_reward": valid_mean("reward"),
        "valid_dynamic_completion": valid_mean("route_completion"),
        "valid_dynamic_collision_rate": valid_mean("collision"),
        "valid_dynamic_arrive_dest_rate": valid_mean("arrive_dest"),
        "valid_dynamic_out_of_road_rate": rate(valid_road_events, len(valid)),
        "valid_dynamic_road_violation_termination_rate": rate(
            valid_road_terminations,
            len(valid),
        ),
    }


def ego_position_2d(env: Any) -> Optional[np.ndarray]:
    ego = getattr(env, "agent", None)
    position = getattr(ego, "position", None)
    if position is None:
        return None
    value = np.asarray(position, dtype=float).reshape(-1)
    return value[:2].copy() if value.size >= 2 else None


def ego_speed_mps(env: Any) -> float:
    ego = getattr(env, "agent", None)
    velocity = getattr(ego, "velocity", None)
    if velocity is None:
        return 0.0
    value = np.asarray(velocity, dtype=float).reshape(-1)
    return float(np.linalg.norm(value[:2])) if value.size >= 2 else 0.0


def control_step_seconds(env: Any, fallback: float = 0.1) -> float:
    config = getattr(env, "config", None)
    if not hasattr(config, "get"):
        return fallback
    physics_step = float(config.get("physics_world_step_size", 0.02))
    decision_repeat = float(config.get("decision_repeat", 5))
    value = physics_step * decision_repeat
    return value if np.isfinite(value) and value > 0.0 else fallback


def speed_stats(speed_samples_mps: Iterable[float]) -> Dict[str, float]:
    values = np.asarray(list(speed_samples_mps), dtype=float)
    if not len(values):
        return {
            "avg_speed_mps": 0.0,
            "avg_speed_kmh": 0.0,
            "speed_p50_mps": 0.0,
            "speed_p95_mps": 0.0,
            "max_speed_mps": 0.0,
            "low_speed_rate": 0.0,
            "stopped_rate": 0.0,
        }
    average = float(values.mean())
    return {
        "avg_speed_mps": average,
        "avg_speed_kmh": 3.6 * average,
        "speed_p50_mps": float(np.percentile(values, 50)),
        "speed_p95_mps": float(np.percentile(values, 95)),
        "max_speed_mps": float(values.max()),
        "low_speed_rate": float(np.mean(values < LOW_SPEED_THRESHOLD_MPS)),
        "stopped_rate": float(np.mean(values < STOPPED_THRESHOLD_MPS)),
    }


def latency_stats(
    latency_samples_ms: Iterable[float],
    deadline_ms: float,
) -> Dict[str, float]:
    values = np.asarray(list(latency_samples_ms), dtype=float)
    deadline_ms = max(float(deadline_ms), 1e-9)
    if not len(values):
        return {
            "rta_filter_calls": 0,
            "rta_filter_mean_ms": 0.0,
            "rta_filter_p50_ms": 0.0,
            "rta_filter_p95_ms": 0.0,
            "rta_filter_p99_ms": 0.0,
            "rta_filter_max_ms": 0.0,
            "rta_deadline_ms": deadline_ms,
            "rta_deadline_miss_rate": 0.0,
            "rta_budget_utilization": 0.0,
        }
    average = float(values.mean())
    return {
        "rta_filter_calls": int(len(values)),
        "rta_filter_mean_ms": average,
        "rta_filter_p50_ms": float(np.percentile(values, 50)),
        "rta_filter_p95_ms": float(np.percentile(values, 95)),
        "rta_filter_p99_ms": float(np.percentile(values, 99)),
        "rta_filter_max_ms": float(values.max()),
        "rta_deadline_ms": deadline_ms,
        "rta_deadline_miss_rate": float(np.mean(values > deadline_ms)),
        "rta_budget_utilization": average / deadline_ms,
    }


def motion_summary(
    episodes: Sequence[Mapping[str, Any]],
    speed_samples_mps: Iterable[float],
) -> Dict[str, float]:
    speed = speed_stats(speed_samples_mps)
    durations = np.asarray(
        [float(item["episode_duration_seconds"]) for item in episodes],
        dtype=float,
    )
    distances = np.asarray(
        [float(item["distance_m"]) for item in episodes],
        dtype=float,
    )
    arrivals = [item for item in episodes if bool(item["arrive_dest"])]
    arrival_durations = np.asarray(
        [float(item["episode_duration_seconds"]) for item in arrivals],
        dtype=float,
    )
    arrival_speeds = np.asarray(
        [float(item["avg_speed_mps"]) for item in arrivals],
        dtype=float,
    )
    total_duration = float(durations.sum()) if len(durations) else 0.0
    result = {
        **speed,
        "avg_distance_speed_mps": (
            float(distances.sum()) / total_duration if total_duration > 0.0 else 0.0
        ),
        "avg_episode_duration_seconds": (
            float(durations.mean()) if len(durations) else 0.0
        ),
        "avg_distance_m": float(distances.mean()) if len(distances) else 0.0,
        "arrived_episode_count": int(len(arrivals)),
        "arrived_avg_time_seconds": (
            float(arrival_durations.mean()) if len(arrival_durations) else 0.0
        ),
        "arrived_p95_time_seconds": (
            float(np.percentile(arrival_durations, 95))
            if len(arrival_durations)
            else 0.0
        ),
        "arrived_avg_speed_mps": (
            float(arrival_speeds.mean()) if len(arrival_speeds) else 0.0
        ),
    }
    return result
