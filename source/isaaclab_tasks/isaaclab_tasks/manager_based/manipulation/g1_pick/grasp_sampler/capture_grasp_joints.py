"""Captures the converged grasp's joint angles (7 arm + 6 hand) after the policy settles,
across many envs, and prints percentile/mean stats -- used as the reference pose for RSI.
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
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--settle_steps", type=int, default=150)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
_HAND_JOINTS = [
    "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint",
    "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint",
]
_PALM_BODY = "right_wrist_yaw_link"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
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
    arm_ids = [robot.joint_names.index(n) for n in _ARM_JOINTS]
    hand_ids = [robot.joint_names.index(n) for n in _HAND_JOINTS]
    palm_id = robot.body_names.index(_PALM_BODY)

    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.settle_steps):
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

    arm_pos = robot.data.joint_pos[:, arm_ids]
    hand_pos = robot.data.joint_pos[:, hand_ids]
    palm_pos_w = robot.data.body_pos_w[:, palm_id]
    cube_pos_w = obj.data.root_pos_w

    print("\n===== CONVERGED GRASP JOINT ANGLES (rad) =====")
    print("arm mean:", arm_pos.mean(dim=0).tolist())
    print("arm median:", arm_pos.median(dim=0).values.tolist())
    print("arm std:", arm_pos.std(dim=0).tolist())
    print("hand mean:", hand_pos.mean(dim=0).tolist())
    print("hand median:", hand_pos.median(dim=0).values.tolist())
    print("hand std:", hand_pos.std(dim=0).tolist())
    print("\npalm_pos_w mean:", palm_pos_w.mean(dim=0).tolist())
    print("cube_pos_w mean:", cube_pos_w.mean(dim=0).tolist())
    print("palm-cube offset mean:", (palm_pos_w - cube_pos_w).mean(dim=0).tolist())

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
