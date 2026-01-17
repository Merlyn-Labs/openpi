import numpy as np
import torch
import os
import json
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi_client.base_policy import BasePolicy
from openpi.policies import policy as _policy
from openpi_client.image_tools import resize_with_pad
from collections import deque
import copy
import traceback

RESIZE_SIZE = 224

def create_policy(ckpt_dir: str) -> _policy.Policy:
    """Create a policy from the given arguments."""
    config = _config.get_config("pi05_b1k_inference_final")
    return _policy_config.create_trained_policy(
        config, ckpt_dir, default_prompt=None
    )

class B1KPolicyWrapper():
    def __init__(
        self,
        policy: BasePolicy,
        config: _config.TrainConfig,
        text_prompt : str = "Turn on the radio receiver that's on the table in the living room.",
    ) -> None:
        self.policy = policy
        self.text_prompt = text_prompt
        self.current_task_id = None
        self.step_counter = 0

        self.configs = {
            "coarse_lower_horizon": {
                "control_mode": "receeding_horizon",
                "max_len": 50,
                "action_horizon": 50,
                "temporal_ensemble_max": 1,
                "exp_k_value": 1.0,
            },
            "coarse": {
                "control_mode": "receeding_horizon",
                "max_len": 100,
                "action_horizon": 100,
                "temporal_ensemble_max": 1,
                "exp_k_value": 1.0,
            },
            "fine": {
                "control_mode": "receeding_temporal",
                "max_len": 72,
                "action_horizon": 12,
                "temporal_ensemble_max": 6,
                "exp_k_value": 0.005,
            },
            "fine_higher_k": {
                "control_mode": "receeding_temporal",
                "max_len": 72,
                "action_horizon": 12,
                "temporal_ensemble_max": 6,
                "exp_k_value": 0.5,
            },
        }

        self.task_idx_ckpt_path_map = {
            16: "openpi_05_20251115_045832/36000",               # moving_boxes_to_storage
            22: "psor_openpi_05_20251116_062730/27000",          # putting_shoes_on_rack
            15: "openpi_05_20251115_071839/15000",               # bringing_in_wood
            42: "chop_an_onion_openpi_05_20251116_220711/9000",  # chop_an_onion
            44: "cw_openpi_05_20251116_072941/15000",            # chopping_wood
            6:  "hee_openpi_05_20251116_064228/18000",           # hiding_Easter_eggs
            13: "ltc_openpi_05_20251116_073405/15000",           # loading_the_car
            1:  "openpi_05_20251115_072623/21000",               # picking_up_trash
            38: "sfb_openpi_05_20251116_065743/24000",           # spraying_for_bugs
            39: "sft_openpi_05_20251116_070631/21000",           # spraying_fruit_trees
            3:  "cupaf_openpi_05_20251116_073015/18000",         # cleaning_up_plates_and_food
            0:  "openpi_05_20251115_050323/9000",                # turning_on_radio
            8:  "rkf_openpi_05_20251116_220634/3000",            # rearranging_kitchen_furniture
            2:  "pahd_openpi_05_20251116_073515/3000",           # putting_away_Halloween_decorations (lower stepcount)
        }

        self.task_idx_config_type_map = {
            0: "coarse",                 # turning_on_radio was eval'd with RH 50, so that's what we'll keep
            15: "coarse",                # bringing_in_wood should be better with RH, but we got 0.0 anyway
            13: "coarse",                # loading_the_car should be better with RH, but we got 0.0 anyway
            16: "coarse_lower_horizon",  # moving_boxes_to_storage was submitted with RH 50, so that's what we'll keep
            22: "fine_higher_k",         # putting_shoes_on_rack was better with receeding_temporal with k=0.5
        }                                # For everything else, we'll default to `fine` which is receeding_temporal with k=0.005

        dataset_root = config.data.base_config.behavior_dataset_root
        self.task_prompt_map = {}
        TASKS_METADATA_PATH = os.path.join(dataset_root, "meta/tasks.jsonl")
        if os.path.exists(TASKS_METADATA_PATH):
            with open(TASKS_METADATA_PATH, "r") as f:
                for line in f:
                    task = json.loads(line)
                    self.task_prompt_map[task["task_index"]] = task["task"]

        # 16 is moving_boxes_to_storage, the "default" task
        self.maybe_set_new_task(16)

    def maybe_set_new_task(self, task_id: int):
        if task_id == self.current_task_id:
            print(f"Task {task_id} already set, skipping")
            return

        print(f"Setting new task {task_id}")
        self.current_task_id = task_id
        config_type = self.get_config_type(task_id)
        self.action_horizon = self.configs[config_type]["action_horizon"]
        self.replan_interval = self.action_horizon
        self.max_len = self.configs[config_type]["max_len"]
        self.temporal_ensemble_max = self.configs[config_type]["temporal_ensemble_max"]
        self.exp_k_value = self.configs[config_type]["exp_k_value"]
        self.control_mode = self.configs[config_type]["control_mode"]
        self.action_queue = deque([], maxlen=self.action_horizon)
        self.last_action = {"actions": np.zeros((self.action_horizon, 23), dtype=np.float64)}
        self.step_counter = 0
        self.ckpt_path = self.get_ckpt_path(task_id)
        self.policy = create_policy(self.ckpt_path)

    def get_config_type(self, task_id: int) -> str:
        return self.task_idx_config_type_map.get(int(task_id), "fine")

    def get_ckpt_path(self, task_id: int) -> str:
        ckpt_sub_dir = self.task_idx_ckpt_path_map.get(int(task_id), "openpi_05_20251113_045215/81000")
        print(f"task_id={task_id}, ckpt_sub_dir={ckpt_sub_dir}")
        ckpt_root = os.environ.get("OPENPI_CKPT_ROOT", "./outputs/checkpoints")
        return f"{ckpt_root}/{ckpt_sub_dir}"

    def reset(self):
        self.action_queue = deque([],maxlen=self.action_horizon)
        self.last_action = {"actions": np.zeros((self.action_horizon, 23), dtype=np.float64)}
        self.step_counter = 0

    def get_prompt_from_obs(self, obs: dict) -> str:
        if "prompt" in obs:
            return obs["prompt"]

        task_id = int(obs["task_id"])
        if task_id in self.task_prompt_map:
            return self.task_prompt_map[task_id]

        return self.text_prompt

    def process_obs(self, obs: dict) -> dict:
        """
        Process the observation dictionary to match the expected input format for the model.
        """
        prop_state = obs["robot_r1::proprio"][None]
        img_obs = np.stack(
            [
                resize_with_pad(
                    obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][None, ..., :3],
                    RESIZE_SIZE,
                    RESIZE_SIZE
                ),
                resize_with_pad(
                    obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"][None, ..., :3], 
                    RESIZE_SIZE,
                    RESIZE_SIZE
                ),
                resize_with_pad(
                    obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"][None, ..., :3],
                    RESIZE_SIZE,
                    RESIZE_SIZE
                ),
            ],
            axis=1,
        )
        processed_obs = {
            "observation": img_obs,  # Shape: (1, 3, H, W, C)
            "proprio": prop_state,
            "prompt": self.get_prompt_from_obs(obs),
            "task_index": int(obs["task_id"]),  # read it in as task_index because that's what _transforms.ExtractTaskID() expects
        }
        return processed_obs
    
    def act_receeding_temporal(self, input_obs):
        # Step 1: check if we should re-run policy
        if self.step_counter % self.replan_interval == 0:
            # Run policy every K steps
            nbatch = copy.deepcopy(input_obs)
            if nbatch["observation"].shape[-1] != 3:
                nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

            joint_positions = nbatch["proprio"][0]
            batch = {
                "observation/egocentric_camera": nbatch["observation"][0, 0],
                "observation/wrist_image_left": nbatch["observation"][0, 1],
                "observation/wrist_image_right": nbatch["observation"][0, 2],
                "observation/state": joint_positions,
                "prompt": input_obs["prompt"],
                "task_index": input_obs["task_index"],
            }

            try:
                action = self.policy.infer(batch)
                self.last_action = action
            except Exception as e:
                action = self.last_action
                print(f"Error in action prediction, using last action: {traceback.format_exc()}")

            target_joint_positions = action["actions"].copy()

            # Add this sequence to action queue
            new_seq = deque([a for a in target_joint_positions[:self.max_len]])
            self.action_queue.append(new_seq)

            # Optional: limit memory
            while len(self.action_queue) > self.temporal_ensemble_max:
                self.action_queue.popleft()

        # Step 2: Smooth across current step from all stored sequences
        if len(self.action_queue) == 0:
            raise ValueError("Action queue empty in receeding_temporal mode.")

        actions_current_timestep = np.empty((len(self.action_queue), self.action_queue[0][0].shape[0]))

        for i in range(len(self.action_queue)):
            actions_current_timestep[i] = self.action_queue[i].popleft()

        # Drop exhausted sequences
        self.action_queue = deque([q for q in self.action_queue if len(q) > 0])

        # Apply temporal ensemble
        exp_weights = np.exp(self.exp_k_value * np.arange(actions_current_timestep.shape[0]))
        exp_weights = exp_weights / exp_weights.sum()

        final_action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)

        # Preserve grippers from most recent rollout
        final_action[-9] = actions_current_timestep[-1, -9]
        final_action[-1] = actions_current_timestep[-1, -1]
        final_action = final_action[None]

        self.step_counter += 1

        return final_action

    def act(self, input_obs):
        # TODO reformat data into the correct format for the model
        # TODO: communicate with justin that we are using numpy to pass the data. Also we are passing in uint8 for images 
        """
        Model input expected: 
            📌 Key: observation/exterior_image_1_left
            Type: ndarray
            Dtype: uint8
            Shape: (224, 224, 3)

            📌 Key: observation/exterior_image_2_left
            Type: ndarray
            Dtype: uint8
            Shape: (224, 224, 3)

            📌 Key: observation/joint_position
            Type: ndarray
            Dtype: float64
            Shape: (16,)

            📌 Key: prompt
            Type: str
            Value: do something
        
        Model will output:
            📌 Key: actions
            Type: ndarray
            Dtype: float64
            Shape: (10, 16)
        """
        # Check if new task and then set it if so
        self.maybe_set_new_task(input_obs["task_id"])

        input_obs = self.process_obs(input_obs)
        if self.control_mode == 'receeding_temporal':
            return torch.from_numpy(self.act_receeding_temporal(input_obs))
        
        if self.control_mode == 'receeding_horizon':
            if len(self.action_queue) > 0:
                # pop the first action in the queue
                final_action = self.action_queue.popleft()[None]
                return torch.from_numpy(final_action)
        
        nbatch = copy.deepcopy(input_obs)
        if nbatch["observation"].shape[-1] != 3: 
            # make B, num_cameras, H, W, C  from B, num_cameras, C, H, W
            # permute if pytorch
            nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

        # nbatch["proprio"] is B, 16, where B=1
        joint_positions = nbatch["proprio"][0]
        batch = {
            "observation/egocentric_camera": nbatch["observation"][0, 0],
            "observation/wrist_image_left": nbatch["observation"][0, 1],
            "observation/wrist_image_right": nbatch["observation"][0, 2],
            "observation/state": joint_positions,
            "prompt": nbatch["prompt"],
            "task_index": nbatch["task_index"],
        }
        # print(f"batch['prompt']: {batch['prompt']}")

        try:
            action = self.policy.infer(batch) 
            self.last_action = action
        except Exception as e:
            action = self.last_action
            raise e
        # convert to absolute action and append gripper command
        # action shape: (10, 23), joint_positions shape: (23,)
        # Need to broadcast joint_positions to match action sequence length
        target_joint_positions = action["actions"].copy() 
        if self.control_mode == 'receeding_horizon':
            # print(f"target_joint_positions shape: {target_joint_positions.shape}")
            self.action_queue = deque([a for a in target_joint_positions[:self.max_len]])
            final_action = self.action_queue.popleft()[None]

        # # temporal emsemble start
        elif self.control_mode == 'temporal_ensemble':
            new_actions = deque(target_joint_positions)
            self.action_queue.append(new_actions)
            actions_current_timestep = np.empty((len(self.action_queue), target_joint_positions.shape[1]))
            
            for i, q in enumerate(self.action_queue):
                actions_current_timestep[i] = q.popleft()

            exp_weights = np.exp(self.exp_k_value * np.arange(actions_current_timestep.shape[0]))
            exp_weights = exp_weights / exp_weights.sum()

            final_action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)
            final_action[-9] = target_joint_positions[0, -9]
            final_action[-1] = target_joint_positions[0, -1]
            final_action = final_action[None]
        else:
            final_action = target_joint_positions
        return torch.from_numpy(final_action)
