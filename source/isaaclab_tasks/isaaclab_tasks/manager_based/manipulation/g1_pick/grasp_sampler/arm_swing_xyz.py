"""Tracks the palm's full (x, y, z) position through a SUSTAINED swing of one arm joint
at a time (not a single-step perturbation), starting from the policy's own converged
grasp pose. Run once per joint-under-test via --swing_joint_idx, --swing_delta.

Usage:
  python arm_swing_xyz.py --checkpoint <path> --headless --device cuda:1 \
      --swing_joint_idx 0 --swing_delta 1.0
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
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--settle_steps", type=int, default=150)
parser.add_argument("--swing_joint_idx", type=int, required=True)
parser.add_argument("--swing_delta", type=float, required=True)
parser.add_argument("--swing_steps", type=int, default=40)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
_PALM_BODY = "right_wrist_yaw_link"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1
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
    palm_id = robot.body_names.index(_PALM_BODY)
    obj = env.unwrapped.scene["target_object"]

    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.settle_steps):
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

    palm_xyz_before = robot.data.body_pos_w[0, palm_id].clone()
    cube_xyz_before = obj.data.root_pos_w[0].clone()
    print(f"[settled] palm xyz (m): {palm_xyz_before.tolist()}")
    print(f"[settled] cube xyz (m): {cube_xyz_before.tolist()}")
    joint_name = _ARM_JOINT_NAMES[args_cli.swing_joint_idx]
    print(f"[swing] joint={joint_name} idx={args_cli.swing_joint_idx} delta={args_cli.swing_delta} steps={args_cli.swing_steps}")

    traj = []
    with torch.inference_mode():
        for sstep in range(args_cli.swing_steps):
            actions = policy(obs)
            actions[0, args_cli.swing_joint_idx] += args_cli.swing_delta
            obs, _, dones, extras = env.step(actions)
            p = robot.data.body_pos_w[0, palm_id].clone()
            traj.append(p.tolist())

    palm_xyz_after = robot.data.body_pos_w[0, palm_id].clone()
    cube_xyz_after = obj.data.root_pos_w[0].clone()
    d = (palm_xyz_after - palm_xyz_before) * 100.0
    dc = (cube_xyz_after - cube_xyz_before) * 100.0

    print(f"\n===== SWING RESULT: {joint_name} (idx {args_cli.swing_joint_idx}), delta={args_cli.swing_delta} =====")
    print(f"palm xyz before (cm): {[round(v*100,3) for v in palm_xyz_before.tolist()]}")
    print(f"palm xyz after  (cm): {[round(v*100,3) for v in palm_xyz_after.tolist()]}")
    print(f"palm d_xyz (cm): dx={d[0].item():.3f} dy={d[1].item():.3f} dz={d[2].item():.3f}")
    print(f"cube d_xyz (cm): dx={dc[0].item():.3f} dy={dc[1].item():.3f} dz={dc[2].item():.3f}")
    print("trajectory (every 5 steps, palm z cm):", [round(traj[i][2]*100,3) for i in range(0, len(traj), 5)])

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
