"""Standalone diagnostic: during a REAL policy rollout (the actual task env, so
InspireMimicAction is active exactly as in training), measures whether the coupled
finger joints (intermediate/distal) are actually tracking their software-computed
targets, or falling short / pinned at a limit -- to find out whether the 2-3cm
fingertip-to-cube gap on index/middle/ring/pinky (every checkpoint, every lineage) is a
tracking/actuator problem or something else. measure_coupling.py tested the PASSIVE
physics-only response (no InspireMimicAction), which is the wrong mechanism to test --
this probe uses the real one.
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
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

_PROX = ["R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint",
         "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint"]
_MIM = ["R_thumb_intermediate_joint", "R_thumb_distal_joint", "R_index_intermediate_joint",
        "R_middle_intermediate_joint", "R_ring_intermediate_joint", "R_pinky_intermediate_joint"]
_MIM_LABEL = ["thumb_int", "thumb_dist", "idx_int", "mid_int", "ring_int", "pinky_int"]


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
    prox_ids = [robot.joint_names.index(n) for n in _PROX]
    mim_ids = [robot.joint_names.index(n) for n in _MIM]

    lo = robot.data.joint_pos_limits[0, mim_ids, 0].cpu().numpy()
    hi = robot.data.joint_pos_limits[0, mim_ids, 1].cpu().numpy()
    prox_lo = robot.data.joint_pos_limits[0, prox_ids, 0].cpu().numpy()
    prox_hi = robot.data.joint_pos_limits[0, prox_ids, 1].cpu().numpy()

    prox_pos, mim_pos, mim_target, mim_err = [], [], [], []

    obs = env.get_observations()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

        prox_pos.append(robot.data.joint_pos[:, prox_ids].cpu().numpy())
        mim_pos.append(robot.data.joint_pos[:, mim_ids].cpu().numpy())
        tgt = robot.data.joint_pos_target[:, mim_ids].cpu().numpy()
        mim_target.append(tgt)
        mim_err.append(tgt - robot.data.joint_pos[:, mim_ids].cpu().numpy())

    prox_pos = np.concatenate(prox_pos)   # (N*T, 6)
    mim_pos = np.concatenate(mim_pos)
    mim_target = np.concatenate(mim_target)
    mim_err = np.concatenate(mim_err)

    n = prox_pos.shape[0]
    ss = slice(n // 3, n)  # steady state (skip early-episode transient)

    print("\n" + "=" * 78)
    print("PROXIMAL joints: steady-state position vs their own limits")
    print("=" * 78)
    for i, name in enumerate(_PROX):
        p = prox_pos[ss, i]
        frac = (p.mean() - prox_lo[i]) / (prox_hi[i] - prox_lo[i] + 1e-9)
        print(f"{name:32s} pos_mean={p.mean():+.3f}  limits=[{prox_lo[i]:+.3f},{prox_hi[i]:+.3f}]"
              f"  frac_of_range_used={frac:.2%}")

    print("\n" + "=" * 78)
    print("MIMIC/COUPLED joints: does actual position track its SOFTWARE-COMMANDED target?")
    print("=" * 78)
    for i, name in enumerate(_MIM_LABEL):
        a = mim_pos[ss, i]
        t = mim_target[ss, i]
        e = mim_err[ss, i]
        frac = (a.mean() - lo[i]) / (hi[i] - lo[i] + 1e-9)
        print(f"{name:12s} actual_mean={a.mean():+.3f}  target_mean={t.mean():+.3f}  "
              f"tracking_err_mean={e.mean():+.3f} (abs={np.abs(e).mean():.3f})  "
              f"limits=[{lo[i]:+.3f},{hi[i]:+.3f}]  frac_of_range_used={frac:.2%}")

    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
