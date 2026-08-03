import pathlib
pathlib.PosixPath = pathlib.WindowsPath

import sys
sys.argv = [
    "bmt/ADV-BMT_dataset_generate.py",
    "--dir", "Baseline Scenarios/scenarionet_waymo_training_500",
    "--save_dir", "generated_scenarios",
    "--TF_mode", "all_TF_except_adv",
    "--num_scenario", "1",
    "--num_mode", "1",
]
exec(open("bmt/ADV-BMT_dataset_generate.py").read())
