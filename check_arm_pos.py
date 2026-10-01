import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg("Isaac-G1-Pick-v0", num_envs=1)
env = gym.make("Isaac-G1-Pick-v0", cfg=env_cfg)
obs, _ = env.reset()
uenv = env.unwrapped
robot = uenv.scene["robot"]

for name in ["torso_link", "right_shoulder_pitch_link", "right_shoulder_roll_link",
             "right_elbow_link", "R_hand_base_link", "pelvis"]:
    if name in robot.body_names:
        idx = robot.body_names.index(name)
        pos = robot.data.body_pos_w[0, idx]
        print(f"[check] {name}: {pos.cpu().numpy()}")
    else:
        print(f"[check] {name}: NOT FOUND")

cube = uenv.scene["target_object"]
print(f"[check] target_object (cube) default pos: {cube.data.root_pos_w[0].cpu().numpy()}")
print(f"[check] robot root pos: {robot.data.root_pos_w[0].cpu().numpy()}")
print(f"[check] all body names: {robot.body_names}")

env.close()
simulation_app.close()
