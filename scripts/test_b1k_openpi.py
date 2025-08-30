from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np
from openpi.policies import b1k_policy
from openpi.shared.eval_b1k_wrapper import OpenPIWrapper

# example = b1k_policy.make_b1k_example()
# print("\n=== Example Contents ===")
# print("-" * 50)
# for key, value in example.items():
#     print(f"\n📌 Key: {key}")
#     print(f"   Type: {type(value).__name__}")
#     if isinstance(value, np.ndarray):
#         print(f"   Dtype: {value.dtype}")
#         print(f"   Shape: {value.shape}")
#     else:
#         print(f"   Value: {value}")
# print("-" * 50 + "\n")

config = config.get_config("pi0_fast_sim_b1k_450")
checkpoint_dir = "/svl/u/ravenh/openpi/openpi_checkpoints/pi0_fast_sim_b1k_450/pi0_fast_sim_b1k_450"

openpi_policy = OpenPIWrapper(
    model_ckpt_folder = checkpoint_dir,
    ckpt_id=29999,
    text_prompt="pick up the green mug",
)

import h5py
path = "/svl/u/mengdixu/b1k-datagen/mimicgen/datasets/demo_450.hdf5"
data = h5py.File(path, "r")
demo_0 = data["data/demo_0"]
obs_ego = demo_0["obs/robot_r1::robot_r1:eyes:Camera:0::rgb"][0,:,:,:3]
obs_wrist_left = demo_0["obs/robot_r1::robot_r1:left_eef_link:Camera:0::rgb"][0,:,:,:3]
obs_wrist_right = demo_0["obs/robot_r1::robot_r1:right_eef_link:Camera:0::rgb"][0,:,:,:3]
obs = np.stack([obs_ego, obs_wrist_left, obs_wrist_right], axis=0)  #(num_cameras, H, W, C) 
obs = obs[None, None]  #(B, T, num_cameras, H, W, C) 
proprio = demo_0["obs/prop_state"][0,:][None,None] # (B, T, 21)
example = {
    "observation": obs,
    "proprio": proprio,
}
action = openpi_policy.act(example)
first_action = {key: value[0] for key, value in action.items()}
first_action = np.concatenate([v for v in first_action.values()])
gt_action  = demo_0["actions"][0,:]
print(gt_action)

