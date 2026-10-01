"""Empirically measures which of the 7 right-arm action dimensions moves the palm's
(and cube's) Z height the most, at the converged Phase 2+3 grasp pose. Uses 7 parallel
envs -- each lets the trained policy settle into its own grasp for N steps, then env i
gets a fixed perturbation added to ONLY action index i (one extra step, policy action
elsewhere), and we measure the resulting delta-z.

Usage:
  python arm_z_sensitivity.py --checkpoint <path/model_450.pt> --headless --device cuda:1
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
parser.add_argument("--perturb_delta", type=float, default=0.5)
parser.add_argument("--perturb_steps", type=int, default=5)
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
    env_cfg.scene.num_envs = 7  # one per arm joint
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
            if step % 30 == 0:
                print(f"[settle] step {step}/{args_cli.settle_steps}", flush=True)

    palm_z_before = robot.data.body_pos_w[:, palm_id, 2].clone()
    cube_z_before = obj.data.root_pos_w[:, 2].clone()
    print("\n[settled] palm_z:", palm_z_before.tolist())
    print("[settled] cube_z:", cube_z_before.tolist())

    with torch.inference_mode():
        for pstep in range(args_cli.perturb_steps):
            actions = policy(obs)
            for i in range(7):
                actions[i, i] += args_cli.perturb_delta
            obs, _, dones, extras = env.step(actions)

    palm_z_after = robot.data.body_pos_w[:, palm_id, 2].clone()
    cube_z_after = obj.data.root_pos_w[:, 2].clone()

    print("\n===== ARM JOINT Z-SENSITIVITY (perturb_delta={} over {} steps) =====".format(
        args_cli.perturb_delta, args_cli.perturb_steps))
    print(f"{'joint':<28}{'d_palm_z (cm)':>16}{'d_cube_z (cm)':>16}")
    dz_palm = (palm_z_after - palm_z_before) * 100.0
    dz_cube = (cube_z_after - cube_z_before) * 100.0
    for i, name in enumerate(_ARM_JOINT_NAMES):
        print(f"{name:<28}{dz_palm[i].item():>16.3f}{dz_cube[i].item():>16.3f}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
