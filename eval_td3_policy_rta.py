"""Evaluate a trained TD3 policy with an optional plug-in runtime safety layer.

The original ``eval_td3_policy.py`` is intentionally left unchanged. Running
that script therefore preserves the native evaluation path exactly.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from eval_td3_policy import build_eval_env, load_td3_model, scalar
from bmt.runtime_assurance.metrics import (
    control_step_seconds,
    ego_position_2d,
    ego_speed_mps,
    latency_stats,
    motion_summary,
    reference_route_length,
    speed_stats,
    termination_outcome,
    termination_summary,
)


def build_rta_plugin(name: str, config_path: Optional[str]):
    if name == "none":
        return None

    from bmt.runtime_assurance.adaptive_plugin import (
        AdaptiveMetaDriveCBFPlugin,
        FSMHOCBFPlugin,
        RearEscapeCBFPlugin,
    )
    from bmt.runtime_assurance.types import CBFConfig

    if name == 'fsm_rear_physical_guard':
        from bmt.runtime_assurance.rear_guard import RearPhysicalGuardPlugin
        root = Path(__file__).resolve().parent
        physical_path = Path(config_path) if config_path else root/'cfgs/rta_rear_physical.json'
        with physical_path.open(encoding='utf-8') as stream:
            physical = CBFConfig.from_mapping(json.load(stream))
        with (root/'cfgs/rta_fsm_hocbf_default.json').open(encoding='utf-8') as stream:
            legacy = CBFConfig.from_mapping(json.load(stream))
        return RearPhysicalGuardPlugin(physical, legacy)

    plugin_types = {
        "fsm_hocbf_qp": FSMHOCBFPlugin,
        "fsm_rear_escape_cbf_qp": RearEscapeCBFPlugin,
        "predictive_escape_cbf2d": AdaptiveMetaDriveCBFPlugin,
    }
    if name not in plugin_types:
        raise ValueError(f"Unknown runtime-assurance plugin: {name}")

    values: Dict[str, Any] = {}
    if config_path:
        with open(Path(config_path).resolve(), "r", encoding="utf-8") as stream:
            values = json.load(stream)
    return plugin_types[name](CBFConfig.from_mapping(values))


def evaluate(
    model,
    env,
    num_episodes: int,
    start_index: int,
    num_scenarios: int,
    deterministic: bool,
    rta_plugin=None,
) -> Tuple[Dict[str, Any], list]:
    episodes = []

    step_seconds = control_step_seconds(env)
    speed_samples_mps = []
    rta_latency_samples_ms = []
    for episode_index in range(num_episodes):
        scenario_index = (start_index + episode_index) % num_scenarios
        observation, info = env.reset(seed=scenario_index)
        if rta_plugin is not None:
            rta_plugin.reset(env)

        done = False
        total_reward = 0.0
        cost_sum = 0.0
        collision = False
        collision_source = None
        steps = 0
        last_info = info
        interventions = 0
        infeasible_steps = 0
        min_vehicle_h = None
        min_road_h = None
        min_effective_road_h = None
        mode_counts = Counter()
        fallback_counts = Counter()
        nominal_road_violation_steps = 0
        qp_solves = 0
        fsm_transitions = 0
        rear_barrier_steps = rear_outside_safe_set_steps = 0

        episode_speeds_mps = []
        episode_latencies_ms = []
        distance_m = 0.0
        previous_position = ego_position_2d(env)
        while not done:
            nominal_action, _ = model.predict(observation, deterministic=deterministic)
            action = nominal_action
            rta_info = None
            if rta_plugin is not None:
                started_ns = time.perf_counter_ns()
                action, rta_info = rta_plugin.filter_action(env, nominal_action)
                latency_ms = (time.perf_counter_ns() - started_ns) / 1e6
                episode_latencies_ms.append(latency_ms)
                rta_latency_samples_ms.append(latency_ms)
                interventions += int(bool(rta_info["rta_intervened"]))
                infeasible_steps += int(not bool(rta_info["rta_hard_feasible"]))
                mode_counts[str(rta_info["rta_mode"])] += 1
                fallback_mode = str(rta_info.get("rta_fallback_mode", "none"))
                fallback_counts[fallback_mode] += 1
                qp_solves += int(rta_info.get("rta_qp_solve_count", 0))
                fsm_transitions += int(bool(rta_info.get("rta_fsm_transitioned", False)))
                rear_barrier_steps += int(rta_info.get("rta_rear_barrier_count", 0) > 0)
                rear_outside_safe_set_steps += int(rta_info.get("rta_rear_outside_safe_set_count", 0) > 0)
                min_vehicle_h = _running_min(min_vehicle_h, rta_info["rta_min_vehicle_h"])
                min_road_h = _running_min(min_road_h, rta_info["rta_min_road_h"])
                min_effective_road_h = _running_min(
                    min_effective_road_h,
                    rta_info.get("rta_min_effective_road_h"),
                )
                nominal_road_h = rta_info.get("rta_min_road_h")
                nominal_road_violation_steps += int(
                    nominal_road_h is not None and float(nominal_road_h) < 0.0
                )

            observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            current_position = ego_position_2d(env)
            if previous_position is not None and current_position is not None:
                distance_m += float(np.linalg.norm(current_position - previous_position))
            previous_position = current_position
            speed_mps = ego_speed_mps(env)
            episode_speeds_mps.append(speed_mps)
            speed_samples_mps.append(speed_mps)
            cost_sum += float(scalar(info.get("cost"), 0.0))
            step_collision = bool(info.get("crash", False))
            if step_collision and not collision:
                collision_source = _classify_collision(env)
            collision = collision or step_collision
            steps += 1
            last_info = info

        route_completion = float(scalar(last_info.get("route_completion"), 0.0))
        duration_seconds = steps * step_seconds
        episode_speed = speed_stats(episode_speeds_mps)
        arrive_dest = bool(last_info.get("arrive_dest", False))
        out_of_road = bool(last_info.get("out_of_road", False))
        route_length = reference_route_length(env)
        degenerate_scenario = route_length is not None and route_length < 2.0
        termination_reason = termination_outcome(
            arrive_dest=arrive_dest,
            out_of_road=out_of_road,
            collision=collision,
        )
        episode = {
            "episode": episode_index,
            "scenario_index": scenario_index,
            "reward": float(scalar(last_info.get("episode_reward"), total_reward)),
            "reward_sum": total_reward,
            "episode_length": steps,
            "route_completion": max(0.0, min(route_completion, 1.0)),
            "cost": float(scalar(last_info.get("cost"), 0.0)),
            "cost_sum": cost_sum,
            "collision": collision,
            "collision_source": collision_source,
            "crash": bool(last_info.get("crash", False)),
            "out_of_road": out_of_road,
            "arrive_dest": arrive_dest,
            "termination_reason": termination_reason,
            "reference_route_length": route_length,
            "degenerate_scenario": degenerate_scenario,
            "control_step_seconds": step_seconds,
            "episode_duration_seconds": duration_seconds,
            "arrival_time_seconds": duration_seconds if arrive_dest else None,
            "distance_m": distance_m,
            **episode_speed,
        }
        if rta_plugin is not None:
            episode.update(
                {
                    "rta_interventions": interventions,
                    "rta_intervention_rate": interventions / max(steps, 1),
                    "rta_infeasible_steps": infeasible_steps,
                    "rta_min_vehicle_h": min_vehicle_h,
                    "rta_min_road_h": min_road_h,
                    "rta_min_effective_road_h": min_effective_road_h,
                    "rta_mode_counts": dict(mode_counts),
                    "rta_fallback_counts": dict(fallback_counts),
                    "rta_nominal_road_violation_steps": (
                        nominal_road_violation_steps
                    ),
                    "rta_nominal_road_violation_rate": (
                        nominal_road_violation_steps / max(steps, 1)
                    ),
                    "rta_qp_solves": qp_solves,
                    "rta_fsm_transitions": fsm_transitions,
                    "rta_rear_barrier_steps": rear_barrier_steps,
                    "rta_rear_outside_safe_set_steps": rear_outside_safe_set_steps,
                    **latency_stats(
                        episode_latencies_ms,
                        deadline_ms=1000.0 * step_seconds,
                    ),
                }
            )
        episodes.append(episode)
        print(
            "episode={episode} scenario={scenario_index} reward={reward:.3f} "
            "cost={cost:.3f} completion={route_completion:.3f} "
            "speed={avg_speed_mps:.3f}m/s duration={episode_duration_seconds:.1f}s "
            "collision={collision} source={collision_source}".format(**episode)
        )

    summary = _summarize(
        episodes,
        rta_enabled=rta_plugin is not None,
        speed_samples_mps=speed_samples_mps,
        rta_latency_samples_ms=rta_latency_samples_ms,
    )
    return summary, episodes


def _summarize(
    episodes: list,
    rta_enabled: bool,
    speed_samples_mps=None,
    rta_latency_samples_ms=None,
) -> Dict[str, Any]:
    def mean(key: str) -> float:
        values = np.asarray([float(item[key]) for item in episodes], dtype=float)
        return float(values.mean()) if len(values) else 0.0

    sources = Counter(item["collision_source"] for item in episodes if item["collision_source"])
    step_seconds = mean("control_step_seconds")
    deadline_ms = 1000.0 * step_seconds
    summary = {
        "num_episodes": len(episodes),
        "reward": mean("reward"),
        "cost": mean("cost"),
        "completion": mean("route_completion"),
        "collision": mean("collision"),
        "cost_sum": mean("cost_sum"),
        "avg_length": mean("episode_length"),
        "terminal_crash_rate": mean("crash"),
        "out_of_road_rate": mean("out_of_road"),
        "arrive_dest_rate": mean("arrive_dest"),
        "collision_sources": dict(sources),
        "rta_enabled": rta_enabled,
        "control_step_seconds": step_seconds,
        **motion_summary(episodes, speed_samples_mps or []),
        **termination_summary(episodes),
    }
    if rta_enabled:
        fallback_counts = Counter()
        for item in episodes:
            fallback_counts.update(item.get("rta_fallback_counts", {}))
        total_steps = sum(int(item["episode_length"]) for item in episodes)
        total_infeasible_steps = sum(
            int(item["rta_infeasible_steps"]) for item in episodes
        )
        total_nominal_road_violation_steps = sum(
            int(item["rta_nominal_road_violation_steps"]) for item in episodes
        )
        summary.update(
            {
                "rta_avg_interventions": mean("rta_interventions"),
                "rta_intervention_rate": mean("rta_intervention_rate"),
                "rta_avg_infeasible_steps": mean("rta_infeasible_steps"),
                "rta_hard_infeasible_step_rate": (
                    total_infeasible_steps / max(total_steps, 1)
                ),
                "rta_nominal_road_violation_step_rate": (
                    total_nominal_road_violation_steps / max(total_steps, 1)
                ),
                "rta_fallback_counts": dict(fallback_counts),
                "rta_avg_qp_solves": mean("rta_qp_solves"),
                "rta_avg_fsm_transitions": mean("rta_fsm_transitions"),
                "rta_rear_barrier_steps": sum(e.get("rta_rear_barrier_steps", 0) for e in episodes),
                "rta_rear_outside_safe_set_steps": sum(e.get("rta_rear_outside_safe_set_steps", 0) for e in episodes),
                **latency_stats(
                    rta_latency_samples_ms or [],
                    deadline_ms=deadline_ms,
                ),
            }
        )
    else:
        summary.update(latency_stats([], deadline_ms=deadline_ms))
    return summary


def _classify_collision(env) -> Optional[str]:
    """Approximate the first vehicle-contact direction for evaluation telemetry."""

    ego = getattr(env, "agent", None)
    engine = getattr(env, "engine", None)
    traffic_manager = getattr(engine, "traffic_manager", None)
    vehicles = list(getattr(traffic_manager, "vehicles", None) or [])
    if ego is None:
        return "unknown"

    candidates = [vehicle for vehicle in vehicles if vehicle is not ego]
    if not candidates:
        return "non_vehicle"
    closest = min(
        candidates,
        key=lambda vehicle: float(
            np.linalg.norm(np.asarray(vehicle.position)[:2] - np.asarray(ego.position)[:2])
        ),
    )
    relative = np.asarray(closest.position, dtype=float)[:2] - np.asarray(ego.position, dtype=float)[:2]
    heading = np.asarray([np.cos(ego.heading_theta), np.sin(ego.heading_theta)], dtype=float)
    left = np.asarray([-heading[1], heading[0]], dtype=float)
    longitudinal = float(relative @ heading)
    lateral = float(relative @ left)
    if abs(longitudinal) >= abs(lateral):
        return "front" if longitudinal >= 0.0 else "rear"
    return "left_side" if lateral >= 0.0 else "right_side"


def _running_min(current: Optional[float], value: Optional[float]) -> Optional[float]:
    if value is None:
        return current
    value = float(value)
    return value if current is None else min(current, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True, help="Path to TD3 checkpoint .zip")
    parser.add_argument("--eval_data_dir", required=True, help="ScenarioNet/MetaDrive scenario directory")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--rta_plugin",
        choices=["none", "fsm_hocbf_qp", "fsm_rear_escape_cbf_qp", "fsm_rear_physical_guard", "predictive_escape_cbf2d"],
        default="fsm_hocbf_qp",
    )
    parser.add_argument("--rta_config", default=None, help="Optional JSON override for CBFConfig")
    parser.add_argument("--out", default=None, help="Optional JSON result path")
    args = parser.parse_args()

    checkpoint = Path(args.ckpt_path).resolve()
    data_directory = Path(args.eval_data_dir).resolve()
    model = load_td3_model(checkpoint, args.device)
    env, num_scenarios = build_eval_env(data_directory, args.horizon, args.start_index)
    plugin = build_rta_plugin(args.rta_plugin, args.rta_config)

    try:
        summary, episodes = evaluate(
            model=model,
            env=env,
            num_episodes=args.num_episodes,
            start_index=args.start_index,
            num_scenarios=num_scenarios,
            deterministic=not args.stochastic,
            rta_plugin=plugin,
        )
    finally:
        env.close()

    result = {
        "checkpoint": str(checkpoint),
        "eval_data_dir": str(data_directory),
        "runtime_assurance": plugin.describe() if plugin is not None else {"name": "none"},
        "summary": summary,
        "episodes": episodes,
    }
    print("\nRTA Evaluation Summary")
    print(json.dumps(summary, indent=2))

    if args.out:
        output = Path(args.out).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
