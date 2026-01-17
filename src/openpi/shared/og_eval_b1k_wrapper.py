import numpy as np
import torch
import os
import json
from openpi.training import config as _config
from openpi_client.base_policy import BasePolicy
from openpi_client.image_tools import resize_with_pad
from collections import deque
import copy
import traceback

RESIZE_SIZE = 224

class OGB1KPolicyWrapper():
    def __init__(
        self, 
        policy: BasePolicy,
        config: _config.TrainConfig,
        text_prompt : str = "Turn on the radio receiver that's on the table in the living room.",
        control_mode : str = "temporal_ensemble",
        action_horizon: int = 10,
        max_len: int = 50,
        temporal_ensemble_max: int = 5,
        exp_k_value: float = 0.005,
    ) -> None:
        self.policy = policy
        self.text_prompt = text_prompt
        self.control_mode = control_mode
        self.action_queue = deque([], maxlen=action_horizon)
        self.last_action = {"actions": np.zeros((action_horizon, 23), dtype=np.float64)}
        self.action_horizon = action_horizon
        self.exp_k_value = exp_k_value
        self.max_len = max_len
        self.temporal_ensemble_max = temporal_ensemble_max
        self.replan_interval = action_horizon # K: replan every 10 steps
        self.step_counter = 0

        dataset_root = config.data.base_config.behavior_dataset_root
        self.task_prompt_map = {}
        TASKS_METADATA_PATH = os.path.join(dataset_root, "meta/tasks.jsonl")
        if os.path.exists(TASKS_METADATA_PATH):
            with open(TASKS_METADATA_PATH, "r") as f:
                for line in f:
                    task = json.loads(line)
                    self.task_prompt_map[task["task_index"]] = task["task"]

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
        print(f"self.exp_k_value: {self.exp_k_value}")
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
