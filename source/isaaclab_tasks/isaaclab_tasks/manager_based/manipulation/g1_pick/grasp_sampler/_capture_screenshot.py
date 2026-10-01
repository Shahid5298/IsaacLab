"""Single-frame screenshot of a g1_pick env layout (no policy, no video) -- for visually
confirming a distractor layout before committing to a training run.

Usage:
  python _capture_screenshot.py --task Isaac-G1-Pick-Stage2-Play-v0 --out /path/to/out.png
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Capture a single-frame screenshot of a g1_pick layout.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
from PIL import Image

from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.eye = (1.3, -0.8, 1.6)
    env_cfg.viewer.lookat = (0.35, 0.0, 0.87)
    env_cfg.viewer.origin_type = "env"

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env.reset()

    frame = None
    for _ in range(90):
        frame = env.render()

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    Image.fromarray(frame).save(args_cli.out)
    print(f"[screenshot] saved to {args_cli.out}", flush=True)

    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
    simulation_app.close()
