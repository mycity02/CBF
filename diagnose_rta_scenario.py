"""Produce a step-by-step HOCBF diagnostic trace for one MetaDrive scenario."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from eval_td3_policy import build_eval_env, load_td3_model
from eval_td3_policy_rta import build_rta_plugin, _classify_collision


def diagnose(model, env, scenario_index, deterministic, plugin):
    observation, reset_info = env.reset(seed=scenario_index)
    if plugin is not None:
        plugin.reset(env)

    trace = []
    done = False
    collision = False
    collision_source = None
    first_collision_step = None
    step = 0

    while not done:
        nominal_action, _ = model.predict(observation, deterministic=deterministic)
        geometry = _geometry_snapshot(env)
        action = np.asarray(nominal_action, dtype=float)
        rta = None
        if plugin is not None:
            action, rta = plugin.filter_action(env, nominal_action)

        observation, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        step_collision = bool(info.get("crash", False))
        if step_collision and not collision:
            first_collision_step = step
            collision_source = _classify_collision(env)
        collision = collision or step_collision

        trace.append(
            {
                "step": step,
                "nominal_action": np.asarray(nominal_action, dtype=float).tolist(),
                "executed_action": np.asarray(action, dtype=float).tolist(),
                "reward": float(reward),
                "collision": step_collision,
                "crash_vehicle": bool(info.get("crash_vehicle", False)),
                "crash_sidewalk": bool(info.get("crash_sidewalk", False)),
                "crash_object": bool(info.get("crash_object", False)),
                "crash_human": bool(info.get("crash_human", False)),
                "crash_building": bool(info.get("crash_building", False)),
                "out_of_road": bool(info.get("out_of_road", False)),
                "geometry": geometry,
                "rta": rta,
            }
        )
        step += 1

    return {
        "scenario_index": scenario_index,
        "reset_info": _json_safe(reset_info),
        "summary": _summarize_trace(
            trace=trace,
            collision=collision,
            collision_source=collision_source,
            first_collision_step=first_collision_step,
        ),
        "trace": trace,
    }


def _geometry_snapshot(env):
    ego = env.agent
    position = np.asarray(ego.position, dtype=float)[:2]
    velocity = np.asarray(ego.velocity, dtype=float)[:2]
    road_heading = float(
        getattr(getattr(ego, "navigation", None), "current_heading_theta_at_long", ego.heading_theta)
    )
    tangent = np.asarray([np.cos(road_heading), np.sin(road_heading)], dtype=float)
    left = np.asarray([-tangent[1], tangent[0]], dtype=float)
    get_objects = getattr(env.engine, "get_objects", None)
    objects = get_objects() if callable(get_objects) else {}
    candidates = list(objects.values()) if isinstance(objects, dict) else list(objects or [])
    allowed_kinds = {"vehicle", "pedestrian", "cyclist"}
    vehicles = [
        other
        for other in candidates
        if str(getattr(other, "metadrive_type", "")).lower() in allowed_kinds
        and all(
            hasattr(other, attribute)
            for attribute in ("name", "position", "velocity", "heading_theta", "LENGTH", "WIDTH")
        )
    ]
    neighbors = []

    for other in vehicles:
        if other is ego:
            continue
        relative = np.asarray(other.position, dtype=float)[:2] - position
        relative_velocity = np.asarray(other.velocity, dtype=float)[:2] - velocity
        longitudinal = float(relative @ tangent)
        lateral = float(relative @ left)
        longitudinal_rate = float(relative_velocity @ tangent)
        half_extent = _longitudinal_half_extent(ego, tangent) + _longitudinal_half_extent(other, tangent)
        bumper_gap = abs(longitudinal) - half_extent
        if longitudinal >= 0.0:
            region = "front"
            closing_speed = max(0.0, -longitudinal_rate)
        else:
            region = "rear"
            closing_speed = max(0.0, longitudinal_rate)
        lateral_overlap = abs(lateral) <= (float(ego.WIDTH) + float(other.WIDTH)) / 2.0 + 0.5
        if not lateral_overlap:
            region = "left_side" if lateral >= 0.0 else "right_side"
        if bumper_gap <= 0.0:
            ttc = 0.0
        elif closing_speed > 1e-6 and lateral_overlap:
            ttc = bumper_gap / closing_speed
        else:
            ttc = None
        neighbors.append(
            {
                "name": str(other.name),
                "kind": str(getattr(other, "metadrive_type", "vehicle")).lower(),
                "region": region,
                "center_longitudinal": longitudinal,
                "center_lateral": lateral,
                "bumper_gap": bumper_gap,
                "closing_speed": closing_speed,
                "ttc": ttc,
                "speed": float(np.linalg.norm(np.asarray(other.velocity, dtype=float)[:2])),
                "lateral_overlap": lateral_overlap,
            }
        )

    neighbors.sort(key=lambda item: abs(item["center_longitudinal"]))
    return {
        "ego_position": position.tolist(),
        "ego_velocity": velocity.tolist(),
        "ego_speed": float(np.linalg.norm(velocity)),
        "ego_heading": float(ego.heading_theta),
        "road_heading": road_heading,
        "road_left_clearance": _finite_or_none(getattr(ego, "dist_to_left_side", None)),
        "road_right_clearance": _finite_or_none(getattr(ego, "dist_to_right_side", None)),
        "neighbors": neighbors,
    }


def _longitudinal_half_extent(vehicle, tangent):
    heading = np.asarray([np.cos(vehicle.heading_theta), np.sin(vehicle.heading_theta)], dtype=float)
    side = np.asarray([-heading[1], heading[0]], dtype=float)
    return (
        float(vehicle.LENGTH) / 2.0 * abs(float(heading @ tangent))
        + float(vehicle.WIDTH) / 2.0 * abs(float(side @ tangent))
    )


def _summarize_trace(trace, collision, collision_source, first_collision_step):
    rta_steps = [item["rta"] for item in trace if item["rta"] is not None]
    gaps = defaultdict(list)
    ttcs = defaultdict(list)
    for item in trace:
        for neighbor in item["geometry"]["neighbors"]:
            gaps[neighbor["region"]].append(float(neighbor["bumper_gap"]))
            if neighbor["ttc"] is not None:
                ttcs[neighbor["region"]].append(float(neighbor["ttc"]))

    summary = {
        "steps": len(trace),
        "collision": collision,
        "collision_source": collision_source,
        "first_collision_step": first_collision_step,
        "min_gap_by_region": {name: min(values) for name, values in gaps.items()},
        "min_ttc_by_region": {name: min(values) for name, values in ttcs.items()},
    }
    if not rta_steps:
        summary["rta_enabled"] = False
        return summary

    barrier_stats = defaultdict(lambda: {"min_h": float("inf"), "min_psi1": float("inf"), "violations": 0})
    for rta in rta_steps:
        for barrier in rta["rta_barriers"]:
            stats = barrier_stats[barrier["name"]]
            stats["min_h"] = min(stats["min_h"], float(barrier["h"]))
            stats["min_psi1"] = min(stats["min_psi1"], float(barrier["psi1"]))
            stats["violations"] += int(float(barrier["residual"]) < -1e-6)

    summary.update(
        {
            "rta_enabled": True,
            "rta_intervention_steps": sum(int(item["rta_intervened"]) for item in rta_steps),
            "rta_infeasible_steps": sum(int(not item["rta_hard_feasible"]) for item in rta_steps),
            "rta_modes": dict(Counter(item["rta_mode"] for item in rta_steps)),
            "barriers": dict(barrier_stats),
        }
    )
    return summary


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--eval_data_dir", required=True)
    parser.add_argument("--scenario_index", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--rta_plugin",
        choices=["none", "fsm_hocbf_qp", "predictive_escape_cbf2d"],
        default="fsm_hocbf_qp",
    )
    parser.add_argument("--rta_config", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = load_td3_model(Path(args.ckpt_path).resolve(), args.device)
    env, _ = build_eval_env(
        Path(args.eval_data_dir).resolve(),
        args.horizon,
        0,
    )
    plugin = build_rta_plugin(args.rta_plugin, args.rta_config)
    try:
        result = diagnose(
            model=model,
            env=env,
            scenario_index=args.scenario_index,
            deterministic=not args.stochastic,
            plugin=plugin,
        )
    finally:
        env.close()

    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved diagnostic trace to {output}")


if __name__ == "__main__":
    main()
