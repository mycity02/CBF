import copy, pickle, pathlib, sys
sys.path.insert(0, '.')
from omegaconf import OmegaConf
from bmt.utils import REPO_ROOT

config = OmegaConf.load(REPO_ROOT / "cfgs" / "0202_midgpt.yaml")
OmegaConf.set_struct(config, False)
config.PREPROCESSING.keep_all_data = True
config.ROOT_DIR = str(REPO_ROOT.resolve())
OmegaConf.set_struct(config, True)

from bmt.dataset.preprocessor import preprocess_scenario_description_for_motionlm as preprocess
from bmt.gradio_ui.plot import plot_gt

# Load the generated scenario
with open('generated_scenarios/sd_10af3d70d93ef629_SCGEN_all_TF_except_adv_mode_0.pkl', 'rb') as f:
    scenario = pickle.load(f)

print('Scenario ID:', scenario.get('id', 'N/A'))
print('Tracks:', len(scenario.get('tracks', {})))

# Preprocess into data_dict format
try:
    data_dict = preprocess(
        scenario=scenario,
        config=config,
        in_evaluation=True,
        keep_all_data=True,
        backward_prediction=False
    )
    print('Preprocessing OK, keys:', len(data_dict))

    # Render
    img = plot_gt(data_dict, save_path='generated_scenario_vis_full.png')
    print('Saved to generated_scenario_vis_full.png')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
