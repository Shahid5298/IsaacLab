"""Standalone diagnostic: measures the exact per-fingertip distance to the cube SURFACE
(not center) over an inference run, to answer "are the fingers actually touching" with a
number instead of eyeballing video. surface_dist <= 0 means the fingertip point is at or
past the cube's surface (real geometric contact/overlap); 0 < surface_dist <= 0.01 means
inside the contact_offset buffer (robot_cfg.py) where PhysX starts engaging the contact
solver but the fingertip center itself hasn't reached the surface; > 0.01 is genuinely
not touching yet.

Usage:
  python contact_check.py --task Isaac-G1-Pick-Empty-Play-v0 --num_envs 4 --steps 300 \
    --checkpoint <path/model_500.pt> --headless --device cuda:1
"""
import argparse
import os
import sys

_RSL_RL_DIR = os.path.join(os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"),
                           "scripts", "reinforcement_learning", "rsl_rl")
sys.path.append(_RSL_RL_DIR)

from isaaclab.app import AppLauncher
import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-G1-Pick-Empty-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

_RIGHT_HAND_BODIES = [
    "right_wrist_yaw_link", "R_thumb_distal", "R_index_intermediate",
    "R_middle_intermediate", "R_ring_intermediate", "R_pinky_intermediate",
]
_FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
_CUBE_HALF_WIDTH = 0.025


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    resume_path = retrieve_file_path(args_cli.checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["target_object"]
    fingertip_ids = [robot.body_names.index(n) for n in _RIGHT_HAND_BODIES[1:]]

    min_surface_dist = None  # (num_envs, 5), running min per finger per env

    obs = env.get_observations()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

        tips = robot.data.body_pos_w[:, fingertip_ids]  # (N,5,3)
        raw = torch.linalg.norm(tips - obj.data.root_pos_w.unsqueeze(1), dim=-1)  # (N,5)
        surface = raw - _CUBE_HALF_WIDTH
        if min_surface_dist is None:
            min_surface_dist = surface.clone()
        else:
            min_surface_dist = torch.minimum(min_surface_dist, surface)

        if step % 50 == 0:
            print(f"[contact] step {step}/{args_cli.steps}", flush=True)

    m = min_surface_dist.cpu().numpy() * 100.0  # cm, (num_envs, 5)
    print("\n===== MIN per-fingertip distance to CUBE SURFACE over the run (cm) =====", flush=True)
    print("(<=0 = real geometric contact/overlap, 0 to 1cm = inside contact_offset buffer, "
          ">1cm = not touching)", flush=True)
    for e in range(m.shape[0]):
        row = "  ".join(f"{name}={m[e,i]:+.2f}cm" for i, name in enumerate(_FINGER_NAMES))
        print(f"env{e}: {row}", flush=True)
    print("\n--- summary across all envs ---", flush=True)
    for i, name in enumerate(_FINGER_NAMES):
        col = m[:, i]
        n_touch = (col <= 0).sum()
        n_buffer = ((col > 0) & (col <= 1.0)).sum()
        print(f"{name}: min={col.min():+.2f}cm mean={col.mean():+.2f}cm  "
              f"({n_touch}/{len(col)} envs actually touching, {n_buffer}/{len(col)} within 1cm buffer)",
              flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
