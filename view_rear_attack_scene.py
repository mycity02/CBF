import argparse
import logging
import time
from pathlib import Path

from metadrive.envs.scenario_env import ScenarioOnlineEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.utils import read_scenario_data

from bmt.rl_train.train.scgen_rear_attack_generator import RearAttackSCGEN_Generator


def get_filenames(folder_path, prefix="sd_"):
    all_files = []
    for file in Path(folder_path).iterdir():
        if file.is_file() and file.name.startswith(prefix):
            all_files.append(str(file))
    return sorted(all_files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Baseline Scenarios/scenarionet_waymo_training_500")
    parser.add_argument("--scenario_index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--speed_delta", type=float, default=3.0)
    parser.add_argument("--rear_offset", type=float, default=None)
    parser.add_argument("--top_down", action="store_true")
    parser.add_argument("--allow_after_scenario_end", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    scenario_files = get_filenames(str(data_dir))
    if not scenario_files:
        raise RuntimeError(f"No scenario files found in {data_dir}")

    scenario_path = scenario_files[args.scenario_index % len(scenario_files)]
    print(f"Loading source scenario: {scenario_path}")
    source_sd = read_scenario_data(scenario_path)

    print("Generating rear-attack scenario in memory...")
    generator = RearAttackSCGEN_Generator(
        rear_offset=args.rear_offset,
        speed_delta=args.speed_delta,
    )
    rear_attack_sd = generator.generate_from_raw_SD(source_sd, track_length=91)
    if rear_attack_sd is None:
        raise RuntimeError("Rear-attack generation returned None for this scenario")

    scenario_length = int(rear_attack_sd.get("length", args.steps))
    render_steps = args.steps if args.allow_after_scenario_end else min(args.steps, max(1, scenario_length - 1))
    if render_steps < args.steps:
        print(f"Capping render steps to scenario length: {render_steps}/{args.steps}")

    env = ScenarioOnlineEnv(
        {
            "use_render": not args.top_down,
            "agent_policy": ReplayEgoCarPolicy,
            "manual_control": False,
            "render_pipeline": False,
            "show_interface": True,
            "show_logo": False,
            "show_fps": True,
            "log_level": logging.CRITICAL,
            "num_scenarios": 1,
            "start_scenario_index": 0,
            "store_map": False,
            "horizon": max(render_steps + 1, 100),
            "decision_repeat": 1,
            "window_size": (1280, 720),
            "crash_vehicle_done": False,
            "out_of_route_done": False,
            "crash_object_done": False,
            "crash_human_done": False,
            "relax_out_of_road_done": False,
            "vehicle_config": {
                "show_navi_mark": True,
                "show_line_to_dest": False,
                "show_dest_mark": False,
                "no_wheel_friction": True,
            },
        }
    )

    try:
        env.set_scenario(rear_attack_sd)
        env.reset(seed=0)
        delay = 1.0 / args.fps if args.fps > 0 else 0
        for _ in range(render_steps):
            _, _, terminated, truncated, info = env.step([0, 0])
            text = {
                "source_index": args.scenario_index,
                "step": env.episode_step,
                "rear_attack": "99999",
            }
            if args.top_down:
                env.render(mode="top_down", window=True, text=text)
            else:
                env.render(text=text)
            if terminated or truncated:
                print(f"Episode done at step {env.episode_step}: {info}")
                break
            if delay:
                time.sleep(delay)
    finally:
        env.close()


if __name__ == "__main__":
    main()

