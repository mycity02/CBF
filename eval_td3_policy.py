import argparse
import json
import logging
from pathlib import Path

import numpy as np

from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.scenario.utils import get_number_of_scenarios

from bmt.rl_train.train.customized_td3 import Closed_Loop_TD3, CustomizedTD3
from bmt.runtime_assurance.metrics import (
    reference_route_length,
    termination_outcome,
    termination_summary,
)


def build_eval_env(data_dir, horizon, start_index=0):
    num_scenarios = get_number_of_scenarios(str(data_dir))
    if num_scenarios <= 0:
        raise RuntimeError(f"No scenarios found in {data_dir}")

    env = ScenarioEnv(
        config=dict(
            store_map=False,
            use_render=False,
            manual_control=False,
            show_interface=False,
            data_directory=str(data_dir),
            agent_policy=EnvInputPolicy,
            start_scenario_index=start_index,
            num_scenarios=num_scenarios,
            sequential_seed=True,
            horizon=horizon,
            reactive_traffic=False,
            no_static_vehicles=True,
            no_light=True,
            crash_vehicle_done=True,
            out_of_route_done=False,
            crash_object_done=False,
            crash_human_done=False,
            relax_out_of_road_done=False,
            log_level=logging.CRITICAL,
        )
    )
    return env, num_scenarios


def load_td3_model(path, device):
    try:
        return Closed_Loop_TD3.load(str(path), device=device)
    except Exception:
        return CustomizedTD3.load(str(path), device=device)


def scalar(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate(model, env, num_episodes, start_index, num_scenarios, deterministic):
    episodes = []

    for ep in range(num_episodes):
        scenario_index = (start_index + ep) % num_scenarios
        obs, info = env.reset(seed=scenario_index)
        done = False
        total_reward = 0.0
        cost_sum = 0.0
        collision = False
        steps = 0
        last_info = info

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            cost_sum += float(scalar(info.get("cost"), 0.0))
            collision = collision or bool(info.get("crash", False))
            steps += 1
            last_info = info

        route_completion = float(scalar(last_info.get("route_completion"), 0.0))
        terminal_cost = float(scalar(last_info.get("cost"), 0.0))
        episode_reward = float(scalar(last_info.get("episode_reward"), total_reward))
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
            "episode": ep,
            "scenario_index": scenario_index,
            "reward": episode_reward,
            "reward_sum": total_reward,
            "episode_length": steps,
            "route_completion": max(0.0, min(route_completion, 1.0)),
            "cost": terminal_cost,
            "cost_sum": cost_sum,
            "collision": collision,
            "crash": bool(last_info.get("crash", False)),
            "out_of_road": out_of_road,
            "arrive_dest": arrive_dest,
            "termination_reason": termination_reason,
            "reference_route_length": route_length,
            "degenerate_scenario": degenerate_scenario,
        }
        episodes.append(episode)
        print(
            "episode={episode} scenario={scenario_index} reward={reward:.3f} "
            "cost={cost:.3f} completion={route_completion:.3f} "
            "collision={collision} cost_sum={cost_sum:.3f}".format(**episode)
        )

    rewards = np.asarray([e["reward"] for e in episodes], dtype=float)
    lengths = np.asarray([e["episode_length"] for e in episodes], dtype=float)
    completions = np.asarray([e["route_completion"] for e in episodes], dtype=float)
    costs = np.asarray([e["cost"] for e in episodes], dtype=float)
    cost_sums = np.asarray([e["cost_sum"] for e in episodes], dtype=float)
    collisions = np.asarray([float(e["collision"]) for e in episodes], dtype=float)
    crashes = np.asarray([float(e["crash"]) for e in episodes], dtype=float)
    out_of_roads = np.asarray([float(e["out_of_road"]) for e in episodes], dtype=float)
    arrivals = np.asarray([float(e["arrive_dest"]) for e in episodes], dtype=float)

    summary = {
        "num_episodes": len(episodes),
        # Paper Table 5 columns.
        "reward": float(rewards.mean()) if len(rewards) else 0.0,
        "cost": float(costs.mean()) if len(costs) else 0.0,
        "completion": float(completions.mean()) if len(completions) else 0.0,
        "collision": float(collisions.mean()) if len(collisions) else 0.0,
        "cost_sum": float(cost_sums.mean()) if len(cost_sums) else 0.0,
        # Extra diagnostics, not paper columns.
        "avg_length": float(lengths.mean()) if len(lengths) else 0.0,
        "terminal_crash_rate": float(crashes.mean()) if len(crashes) else 0.0,
        "out_of_road_rate": float(out_of_roads.mean()) if len(out_of_roads) else 0.0,
        "arrive_dest_rate": float(arrivals.mean()) if len(arrivals) else 0.0,
        **termination_summary(episodes),
    }
    return summary, episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True, help="Path to TD3 checkpoint .zip")
    parser.add_argument("--eval_data_dir", required=True, help="ScenarioNet/MetaDrive scenario directory")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy actions")
    parser.add_argument("--out", default=None, help="Optional JSON result path")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path).resolve()
    data_dir = Path(args.eval_data_dir).resolve()

    model = load_td3_model(ckpt_path, args.device)
    env, num_scenarios = build_eval_env(data_dir, args.horizon, args.start_index)

    try:
        summary, episodes = evaluate(
            model=model,
            env=env,
            num_episodes=args.num_episodes,
            start_index=args.start_index,
            num_scenarios=num_scenarios,
            deterministic=not args.stochastic,
        )
    finally:
        env.close()

    result = {
        "checkpoint": str(ckpt_path),
        "eval_data_dir": str(data_dir),
        "summary": summary,
        "episodes": episodes,
    }

    print("\nPaper-aligned Summary")
    print("Reward: {reward}".format(**summary))
    print("Cost: {cost}".format(**summary))
    print("Completion: {completion}".format(**summary))
    print("Collision: {collision}".format(**summary))
    print("Cost Sum: {cost_sum}".format(**summary))
    print("\nExtra")
    for key in ["num_episodes", "avg_length", "terminal_crash_rate", "out_of_road_rate", "arrive_dest_rate"]:
        print(f"{key}: {summary[key]}")

    if args.out:
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
