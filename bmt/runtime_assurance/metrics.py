from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


LOW_SPEED_THRESHOLD_MPS = 1.0
STOPPED_THRESHOLD_MPS = 0.1


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
