import argparse
import copy
import os
import pickle

from tqdm import tqdm


def get_filenames(folder_path, prefix="sd_"):
    all_files = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if os.path.isfile(path) and file.startswith(prefix):
            all_files.append(path)
    return sorted(all_files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="ScenarioNet folder containing source .pkl files.")
    parser.add_argument("--num_scenario", type=int, default=1)
    parser.add_argument("--save_dir", required=True, help="Directory to save rear-attack .pkl scenarios.")
    parser.add_argument("--TF_mode", type=str, default="rear_attack")
    parser.add_argument("--num_mode", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="bmt/ckpt/last.ckpt")
    parser.add_argument("--rear_offset", type=float, default=None)
    parser.add_argument("--speed_delta", type=float, default=3.0)
    parser.add_argument("--heading_noise_std", type=float, default=0.0)
    parser.add_argument("--attack_time", type=float, default=None,
                        help="Contact target time in seconds, aligned to 0.5 s; default: final valid SDC frame.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from bmt.rl_train.train.scgen_rear_attack_generator import RearAttackSCGEN_Generator
    import random
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    generator = RearAttackSCGEN_Generator(
        ckpt_path=args.ckpt_path,
        rear_offset=args.rear_offset,
        speed_delta=args.speed_delta,
        heading_noise_std=args.heading_noise_std,
        attack_time=args.attack_time,
    )

    all_scenario_files = get_filenames(args.dir)
    if not all_scenario_files:
        raise RuntimeError(f"No scenario files found in {args.dir}")

    for scenario_idx in tqdm(range(args.num_scenario)):
        source_index = args.start_index + scenario_idx
        if source_index < 0 or source_index >= len(all_scenario_files):
            raise ValueError("Requested scenario range is outside the source dataset")
        sd_path = all_scenario_files[source_index]
        with open(sd_path, "rb") as f:
            scenario_data = pickle.load(f)

        sid = scenario_data["id"]
        for mode_idx in range(args.num_mode):
            rear_attack_sd = generator.generate_from_raw_SD(
                scenario_data=copy.deepcopy(scenario_data),
                track_length=91,
            )
            if rear_attack_sd is None:
                break

            out_name = f"sd_{sid}_SCGEN_{args.TF_mode}_rear_attack_mode_{mode_idx}.pkl"
            with open(os.path.join(args.save_dir, out_name), "xb") as f:
                pickle.dump(rear_attack_sd, f)


if __name__ == "__main__":
    main()
