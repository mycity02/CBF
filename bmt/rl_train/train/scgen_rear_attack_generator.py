import copy

import numpy as np
import torch

from bmt.utils import utils
from bmt.rl_train.train.scgen_generator import (
    SCGEN_Generator,
    _expand_batch,
    _get_mode,
    batch_data,
    is_sdc_parking,
    overwrite_new_sdc_traj_to_SD,
    overwrite_to_scenario_description_new_agent,
    transform_to_global_coordinate,
)


def create_rear_attack_adv_batched(
    batched_data_dict,
    rear_offset=None,
    speed_delta=3.0,
    heading_noise_std=0.0,
):
    """Append a new adversarial vehicle behind SDC at the SDC final valid step."""
    B, T, _, _ = batched_data_dict["decoder/agent_position"].shape
    ego_id = batched_data_dict["decoder/sdc_index"]
    device = ego_id.device
    batch_idx = torch.arange(B, device=device)

    ego_pos = batched_data_dict["decoder/agent_position"][batch_idx, :, ego_id]
    ego_heading = batched_data_dict["decoder/agent_heading"][batch_idx, :, ego_id]
    ego_velocity = batched_data_dict["decoder/agent_velocity"][batch_idx, :, ego_id]
    ego_shape = batched_data_dict["decoder/agent_shape"][batch_idx, :, ego_id]
    ego_mask = batched_data_dict["decoder/agent_valid_mask"][batch_idx, :, ego_id]

    ego_last_valid_step = ego_mask.float().cumsum(dim=1).argmax(dim=1)
    adv_last_valid_step = ego_last_valid_step.clone()

    adv_mask = torch.zeros_like(ego_mask)
    adv_traj = torch.zeros_like(ego_pos)
    adv_heading = torch.zeros_like(ego_heading)
    adv_velocity = torch.zeros_like(ego_velocity)
    adv_shape = torch.zeros_like(ego_shape)

    for b in range(B):
        adv_mask[b, : adv_last_valid_step[b] + 1] = 1

    ego_last_pos = ego_pos[batch_idx, adv_last_valid_step]
    ego_last_xy = ego_last_pos[..., :2]
    ego_last_heading = ego_heading[batch_idx, adv_last_valid_step]
    ego_last_speed = torch.norm(ego_velocity[batch_idx, adv_last_valid_step], dim=-1)
    ego_last_length = ego_shape[batch_idx, adv_last_valid_step, 0]

    forward = torch.stack([torch.cos(ego_last_heading), torch.sin(ego_last_heading)], dim=-1)
    if rear_offset is None:
        # Slightly less than one vehicle length creates a rear-end contact anchor.
        rear_center_offset = torch.clamp(ego_last_length * 0.85, min=3.0, max=5.0)
    else:
        rear_center_offset = torch.full((B,), float(rear_offset), device=device)

    adv_last_pos = adv_traj[batch_idx, adv_last_valid_step]
    adv_last_pos[..., :2] = ego_last_xy - forward * rear_center_offset[:, None]
    if adv_last_pos.shape[-1] > 2:
        adv_last_pos[..., 2:] = ego_last_pos[..., 2:]
    adv_traj[batch_idx, adv_last_valid_step] = adv_last_pos
    adv_heading_value = ego_last_heading + torch.randn(B, device=device) * float(heading_noise_std)
    adv_heading[batch_idx, adv_last_valid_step] = adv_heading_value

    adv_speed = torch.clamp(ego_last_speed + float(speed_delta), min=0.0)
    adv_velocity[batch_idx, adv_last_valid_step] = torch.stack(
        [adv_speed * torch.cos(adv_heading_value), adv_speed * torch.sin(adv_heading_value)], dim=-1
    )
    adv_shape[:, :, :] = ego_shape[batch_idx, adv_last_valid_step].unsqueeze(1).expand(-1, T, -1)

    def append_agent(tensor, new_agent):
        return torch.cat([tensor, new_agent.unsqueeze(2)], dim=2)

    batched_data_dict["decoder/agent_position"] = append_agent(batched_data_dict["decoder/agent_position"], adv_traj)
    batched_data_dict["decoder/agent_heading"] = append_agent(batched_data_dict["decoder/agent_heading"], adv_heading)
    batched_data_dict["decoder/agent_velocity"] = append_agent(batched_data_dict["decoder/agent_velocity"], adv_velocity)
    batched_data_dict["decoder/agent_shape"] = append_agent(batched_data_dict["decoder/agent_shape"], adv_shape)
    batched_data_dict["decoder/agent_valid_mask"] = append_agent(
        batched_data_dict["decoder/agent_valid_mask"], adv_mask
    )

    adv_id = batched_data_dict["decoder/agent_position"].shape[2] - 1

    def append_field(field, value):
        if value.ndim == field.ndim - 1:
            value = value.unsqueeze(1)
        return torch.cat([field, value], dim=1)

    batched_data_dict["decoder/current_agent_shape"] = append_field(
        batched_data_dict["decoder/current_agent_shape"],
        batched_data_dict["decoder/current_agent_shape"][batch_idx, ego_id],
    )
    batched_data_dict["decoder/current_agent_position"] = append_field(
        batched_data_dict["decoder/current_agent_position"],
        batched_data_dict["decoder/current_agent_position"][batch_idx, ego_id],
    )
    batched_data_dict["decoder/agent_type"] = append_field(
        batched_data_dict["decoder/agent_type"],
        batched_data_dict["decoder/agent_type"][batch_idx, ego_id],
    )
    batched_data_dict["decoder/agent_id"] = append_field(
        batched_data_dict["decoder/agent_id"],
        torch.full((B, 1), adv_id, dtype=torch.long, device=device),
    )
    batched_data_dict["decoder/object_of_interest_id"] = append_field(
        batched_data_dict["decoder/object_of_interest_id"],
        torch.full((B, 1), adv_id, dtype=torch.long, device=device),
    )
    batched_data_dict["decoder/current_agent_valid_mask"] = append_field(
        batched_data_dict["decoder/current_agent_valid_mask"],
        torch.ones((B, 1), dtype=torch.long, device=device),
    )

    print("====================================")
    print(f"Rear-attack ADV is created at steps {adv_last_valid_step.tolist()}, ID: {adv_id}")
    print("====================================")
    return batched_data_dict, adv_id, adv_last_valid_step


class RearAttackSCGEN_Generator(SCGEN_Generator):
    def __init__(
        self,
        model_name="0202_midgpt",
        TF_mode="all_TF_except_adv",
        ckpt_path="bmt/ckpt/last.ckpt",
        rear_offset=None,
        speed_delta=3.0,
        heading_noise_std=0.0,
    ):
        super().__init__(model_name=model_name, TF_mode=TF_mode, ckpt_path=ckpt_path)
        self.rear_offset = rear_offset
        self.speed_delta = speed_delta
        self.heading_noise_std = heading_noise_std

    def _create_rear_attack_batch(self, batched_data_dict):
        return create_rear_attack_adv_batched(
            batched_data_dict,
            rear_offset=self.rear_offset,
            speed_delta=self.speed_delta,
            heading_noise_std=self.heading_noise_std,
        )

    def _select_rear_attack_mode(self, output_data, adv_id):
        fallback_mode, _ = self.reject_multi_batch(output_data, self.num_modes)

        for i in range(self.num_modes):
            mask = output_data["decoder/reconstructed_valid_mask"][i, :, adv_id]
            if not bool(mask.any()):
                continue

            last_step = mask.nonzero(as_tuple=True)[0][-1].item()
            ego_id = int(output_data["decoder/sdc_index"][i].item())
            ego_pos = output_data["decoder/agent_position"][i, last_step, ego_id, :2]
            ego_heading = output_data["decoder/agent_heading"][i, last_step, ego_id]
            adv_pos = output_data["decoder/reconstructed_position"][i, last_step, adv_id, :2]
            adv_heading = output_data["decoder/reconstructed_heading"][i, last_step, adv_id]

            forward = torch.stack([torch.cos(ego_heading), torch.sin(ego_heading)])
            lateral_axis = torch.stack([-torch.sin(ego_heading), torch.cos(ego_heading)])
            rel = adv_pos - ego_pos
            longitudinal = torch.dot(rel, forward).item()
            lateral = abs(torch.dot(rel, lateral_axis).item())
            heading_diff = abs(torch.atan2(torch.sin(adv_heading - ego_heading), torch.cos(adv_heading - ego_heading)).item())

            if longitudinal < -0.5 and lateral < 3.0 and heading_diff < np.deg2rad(45):
                return i

        return fallback_mode

    def _generate_from_scenario(self, scenario_data, track_length=91, adaptive=True):
        from bmt.dataset.preprocessor import preprocess_scenario_description_for_motionlm

        if adaptive:
            sid = scenario_data["id"]
            sdc_traj = self.storage[sid].get("SDC_traj")
            if sdc_traj.shape[0] <= 10:
                print("SDC traj length is too short, please check the scenario data. Skipping editing this scenario.")
                return None

            sdc_heading = self.storage[sid].get("SDC_heading")
            sdc_vel = self.storage[sid].get("SDC_vel")
            if isinstance(sdc_traj, list):
                sdc_traj = np.array(sdc_traj)
                sdc_vel = np.array(sdc_vel)
                sdc_heading = np.array(sdc_heading)

            if self.no_adaptive:
                overwritten_sd = copy.deepcopy(scenario_data)
            else:
                overwritten_sd = overwrite_new_sdc_traj_to_SD(
                    copy.deepcopy(scenario_data),
                    sdc_traj,
                    sdc_heading,
                    sdc_vel,
                    track_length=track_length,
                )
        else:
            overwritten_sd = copy.deepcopy(scenario_data)

        original_data_dict = preprocess_scenario_description_for_motionlm(
            scenario=overwritten_sd,
            config=self.config,
            in_evaluation=True,
            keep_all_data=True,
            backward_prediction=True,
            tokenizer=self.pl_model.model.tokenizer,
        )

        data_dict = copy.deepcopy(original_data_dict)
        batched_data_dict = batch_data(utils.numpy_to_torch(data_dict, device=self.pl_model.device))
        batched_data_dict = _expand_batch(batched_data_dict, batch_size=self.num_modes)
        batched_data_dict, adv_id, _ = self._create_rear_attack_batch(batched_data_dict)
        batched_data_dict = {
            k: v.to(self.pl_model.device) if isinstance(v, torch.Tensor) else v
            for k, v in batched_data_dict.items()
        }

        tok_data_dict, _ = self.tokenizer.tokenize(batched_data_dict, backward_prediction=True)
        batched_data_dict.update(tok_data_dict)
        batched_data_dict["in_backward_prediction"] = torch.tensor(
            [True] * self.num_modes, dtype=bool, device=self.pl_model.device
        )

        if batched_data_dict is None:
            print(f"Warning: No rear-attack ADV is generated in {scenario_data['id']} !!!!!")
            return None

        self.adv_id = adv_id
        not_tf_ids = torch.tensor([adv_id]).to(self.pl_model.device)
        output_data = self.GPT_backwardAR(batched_data_dict, not_tf_ids=not_tf_ids)
        best_mode_index = self._select_rear_attack_mode(output_data, adv_id)

        output_data_all_modes = {
            k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in output_data.items()
        }
        output_dict_mode = _get_mode(output_data_all_modes, best_mode_index, num_modes=self.num_modes)
        output_dict_mode_pred = transform_to_global_coordinate(output_dict_mode)

        return overwrite_to_scenario_description_new_agent(
            output_dict_mode=output_dict_mode_pred,
            original_SD=copy.deepcopy(scenario_data),
            adv_id=adv_id,
            from_GT=False,
            ooi=[adv_id],
        )

    def generate(self, scenario_data=None, track_length=91):
        if scenario_data is None:
            assert self.env is not None
            scenario_data = self.env.engine.data_manager.current_scenario
        return self._generate_from_scenario(scenario_data, track_length=track_length, adaptive=True)

    def generate_from_raw_SD(self, scenario_data=None, track_length=91):
        assert scenario_data is not None
        if is_sdc_parking(scenario_data):
            return None
        return self._generate_from_scenario(scenario_data, track_length=track_length, adaptive=False)

