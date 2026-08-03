import argparse
import logging
from pathlib import Path

from PIL import Image

from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.utils import get_number_of_scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Baseline Scenarios/CAT_scenarios")
    parser.add_argument("--scenario_index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--out", default="metadrive_render.png")
    parser.add_argument("--size", type=int, default=1200)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    num_scenarios = get_number_of_scenarios(str(data_dir))
    if num_scenarios <= 0:
        raise RuntimeError(f"No scenarios found in {data_dir}")

    env = ScenarioEnv(
        {
            "use_render": False,
            "agent_policy": ReplayEgoCarPolicy,
            "manual_control": False,
            "render_pipeline": False,
            "show_interface": False,
            "show_logo": False,
            "show_fps": False,
            "log_level": logging.CRITICAL,
            "num_scenarios": num_scenarios,
            "start_scenario_index": args.scenario_index,
            "data_directory": str(data_dir),
            "store_map": False,
            "horizon": max(args.steps + 1, 100),
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
        frame = None
        for _ in range(args.steps):
            env.step([0, 0])
            frame = env.render(
                mode="top_down",
                window=False,
                semantic_map=True,
                target_vehicle_heading_up=False,
                film_size=(args.size, args.size),
                text={"scenario": args.scenario_index, "step": env.episode_step},
            )

        if frame is None:
            raise RuntimeError("MetaDrive did not return a rendered frame")

        out = Path(args.out).resolve()
        Image.fromarray(frame).save(out)
        print(f"Saved MetaDrive render to {out}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
