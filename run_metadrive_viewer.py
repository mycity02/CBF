import argparse
import logging
import time
from pathlib import Path

from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.utils import get_number_of_scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Baseline Scenarios/CAT_scenarios")
    parser.add_argument("--scenario_index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--top_down", action="store_true", help="Use a live top-down window instead of the 3D camera")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    num_scenarios = get_number_of_scenarios(str(data_dir))
    if num_scenarios <= 0:
        raise RuntimeError(f"No scenarios found in {data_dir}")

    env = ScenarioEnv(
        {
            "use_render": True,
            "agent_policy": ReplayEgoCarPolicy,
            "manual_control": False,
            "render_pipeline": False,
            "show_interface": True,
            "show_logo": False,
            "show_fps": True,
            "log_level": logging.CRITICAL,
            "num_scenarios": num_scenarios,
            "start_scenario_index": args.scenario_index,
            "data_directory": str(data_dir),
            "store_map": False,
            "horizon": max(args.steps + 1, 100),
            "decision_repeat": 1,
            "window_size": (1280, 720),
            "vehicle_config": {
                "show_navi_mark": True,
                "show_line_to_dest": False,
                "show_dest_mark": False,
                "no_wheel_friction": True,
            },
        }
    )

    try:
        env.reset(seed=args.scenario_index)
        delay = 1.0 / args.fps if args.fps > 0 else 0
        for _ in range(args.steps):
            env.step([0, 0])
            text = {"scenario": args.scenario_index, "step": env.episode_step}
            if args.top_down:
                env.render(mode="top_down", window=True, text=text)
            else:
                env.render(text=text)
            if delay:
                time.sleep(delay)
    finally:
        env.close()


if __name__ == "__main__":
    main()
